"""Play Mortal against other people's bots on RiichiLab (riichi.dev).

    pip install websockets
    export RIICHI_BOT_TOKEN=...                 # from riichi.dev/bots
    python riichi_lab.py                        # validation ladder
    python riichi_lab.py --ranked               # ranked, once validated
    python riichi_lab.py --ranked --games 50 --log-dir logs/riichi

The platform speaks mjai, which is the language Mortal already thinks in, so
this is mostly plumbing: the server sends the same event stream `mortal.py`
reads from stdin, and `libriichi.mjai.Bot` answers it. Two things are not the
same as a local mjai pipe, and they are what most of this file is about.

First, the server decouples "here is what happened" from "it is your turn":
events arrive as they happen, and a separate `request_action` asks for the
answer and must be echoed back by `request_id`. Bot.react() produces its
reaction when it reads the event, so the reaction is held until the request
for it arrives.

Second, an illegal action is not an error, it is a chombo -- a mangan off the
score and the kyoku over. So nothing is sent that is not one of the actions the
server itself listed, and when Mortal's choice is not among them the fallback
gives up the decision rather than the hanchan.

The `observation` field on `request_action` is a RiichiEnv state snapshot, for
bots that want the platform to do the state tracking. Mortal keeps its own from
the event stream, so it is ignored.
"""
import prelude                                          # noqa: F401

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from os import path

import torch

from engine import MortalEngine
from model import Brain, DQN

# Only for the default checkpoint, and only if a training config happens to be
# here. Playing needs a checkpoint and nothing else -- the shape of the network
# is inside it -- so somewhere with just the weights and libriichi, which is
# the sensible place to run a bot from, `--state-file` is the whole setup.
try:
    from config import config
    DEFAULT_STATE_FILE = config['control']['state_file']
except Exception:
    DEFAULT_STATE_FILE = None

# Sent to say what happened. Everything here goes to the bot verbatim: libriichi
# tracks the table from these and nothing else.
MJAI_EVENTS = frozenset([
    'start_game', 'start_kyoku', 'tsumo', 'dahai', 'chi', 'pon', 'kan',
    'daiminkan', 'ankan', 'kakan', 'reach', 'reach_accepted', 'dora', 'hora',
    'ryukyoku', 'end_kyoku', 'end_game',
])

# Sent to ask for something, or to say nothing at all.
CONTROL_EVENTS = frozenset(['request_action', 'action_ack', 'error'])

# The verdict on a validation game, and the reason to have played it. The
# platform closes the connection right after, so this is the last thing said
# and the only place the answer appears; its fields are the server's business,
# so it is printed whole rather than picked apart against a guess.
VERDICT_EVENTS = frozenset(['validation_result'])


def load_bot_engine(state_file, device, trust=False):
    """Mortal's engine and a tag for it, from a checkpoint on disk.

    `enable_quick_eval` is on: it skips the network where the rules leave one
    move, which is most of a hanchan and all of the time budget worth saving.
    The agari guard is on because the released bot has it on, and a model
    measured without it is a player nobody runs.
    """
    # `weights_only=True` is worth keeping -- a checkpoint is a pickle and this
    # one may have come off a hub -- but older ones wrote `best_perf` as numpy
    # scalars, which it refuses by default. Allowing that one type back is a
    # much smaller hole than turning the check off.
    try:
        state = torch.load(state_file, weights_only=True, map_location='cpu')
    except Exception:
        # Checkpoints from before 2025 stored `best_perf` as numpy scalars,
        # which `weights_only=True` refuses, and the allowlist cannot take them
        # back because numpy 2 moved the class the pickle names. The safe
        # default stays the default: a checkpoint is a pickle, and one off a
        # hub can run anything it likes while it loads. Passing
        # --trust-checkpoint says you know where this one came from.
        if not trust:
            raise SystemExit(
                f'{state_file} will not load under weights_only=True, which is '
                'usually an older checkpoint holding numpy scalars. Loading it '
                'the other way runs whatever the file says to run, so pass '
                '--trust-checkpoint only for a file you produced yourself.')
        logging.warning('loading %s with weights_only=False, as asked', state_file)
        state = torch.load(state_file, weights_only=False, map_location='cpu')
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    num_blocks = cfg['resnet']['num_blocks']
    conv_channels = cfg['resnet']['conv_channels']

    mortal = Brain(version=version, num_blocks=num_blocks,
                   conv_channels=conv_channels).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    tag = state.get('tag')
    if not tag:
        when = datetime.fromtimestamp(state['timestamp'], tz=timezone.utc)
        tag = f'mortal{version}-b{num_blocks}c{conv_channels}-t{when:%y%m%d%H}'

    engine = MortalEngine(
        mortal, dqn,
        version = version,
        is_oracle = False,
        device = device,
        enable_amp = False,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = True,
        name = 'mortal',
    )
    return engine, tag, state.get('steps'), state.get('best_perf')


def _same_action(reaction, candidate):
    """Is `reaction` the action the server is offering as `candidate`?

    Compared on the fields the server names and no others: it lists a discard
    as `{"type": "dahai", "pai": "1m"}`, while Mortal answers with the actor and
    whether it was the drawn tile as well, and those extra fields are not a
    disagreement. A red five is its own tile here -- `5mr` is not `5m` -- so
    the strings are compared as they come.
    """
    if reaction.get('type') != candidate.get('type'):
        return False
    for key, value in candidate.items():
        if key in ('type', 'actor'):
            continue
        mine = reaction.get(key)
        # `consumed` is which tiles the meld is made of, and a meld is not
        # ordered: an ankan offered as ["5m","5m","5m","5mr"] is the one Mortal
        # asked for as ["5mr","5m","5m","5m"]. Compared in order, every kan and
        # every pon whose spelling differed was refused and thrown away on the
        # fallback. Sorted keeps the red five distinct, which it must be -- a
        # pon of three plain fives is a different call from one holding the red.
        if isinstance(value, list) and isinstance(mine, list):
            if sorted(value) != sorted(mine):
                return False
        elif mine != value:
            return False
    return True


def choose(reaction, possible, seat, drawn=None):
    """What to send: Mortal's choice if the server offers it, else a safe one.

    An action the server did not list is a chombo -- a mangan and the kyoku
    ends -- so this never sends one. Mortal disagreeing with the offered list
    means the two disagree about the table, which is worth a loud log and a
    forfeited decision, not a forfeited game: `none` gives up the call, and a
    discard gives up the turn, and both leave the hanchan running.

    The discard it gives up with is the tile just drawn, when that is on offer.
    Taking the first tile the server happens to list is how a hand that was one
    tile from complete gets taken apart to save a single decision; tsumogiri
    leaves the hand exactly as it was, and is what the platform itself plays
    when a bot runs out of time.
    """
    if reaction is not None:
        for candidate in possible:
            if _same_action(reaction, candidate):
                return dict(candidate, actor=seat), None
        # Both sides, in full: a type that is in the list and still did not
        # match means the two disagree about the fields, and only the values
        # show which.
        same_type = [c for c in possible if c.get('type') == reaction.get('type')]
        why = (f'wanted {json.dumps(reaction, default=str)}; '
               + (f'the server offered {json.dumps(same_type)} under that type'
                  if same_type else
                  f'nothing of that type was offered, only '
                  f'{sorted({c.get("type") for c in possible})}'))
    else:
        why = 'no reaction was held for this request'

    for candidate in possible:
        if candidate.get('type') == 'none':
            return dict(candidate, actor=seat), why
    for candidate in possible:
        if candidate.get('type') == 'dahai' and candidate.get('pai') == drawn:
            return dict(candidate, actor=seat, tsumogiri=True), why
    for candidate in possible:
        if candidate.get('type') == 'dahai':
            return dict(candidate, actor=seat), why
    if possible:
        return dict(possible[0], actor=seat), why
    return {'type': 'none', 'actor': seat}, why


class Session:
    """One connection: the events it saw, the games it played, what it scored."""

    def __init__(self, engine, log_dir=None):
        self.engine = engine
        self.log_dir = log_dir
        self.bot = None
        self.seat = None
        self.reaction = None        # held from the event until the request
        self.drawn = None           # this seat's newest draw, for the fallback
        self.lines = []             # this game's events, as they arrived
        self.games = 0
        self.results = []           # (placement, own score) per game
        self.slowest = 0.0
        self.fallbacks = 0
        self.verdicts = []          # whatever the platform said about the games

    def on_mjai(self, line, event):
        """Feed one event to the bot and hold whatever it answers."""
        from libriichi.mjai import Bot

        kind = event['type']
        if kind == 'start_game':
            # The seat is in this message, and Bot wants it at construction, so
            # the bot for a game cannot exist before the game does.
            self.seat = event.get('id', self.seat or 0)
            self.bot = Bot(self.engine, self.seat)
            self.lines = []
            self.reaction = None

        if self.bot is None:
            logging.warning('%s before start_game; ignored', kind)
            return
        self.lines.append(line)

        started = time.perf_counter()
        try:
            answer = self.bot.react(line)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            # BaseException: a state libriichi will not accept makes the Rust
            # side panic, and pyo3 raises that as a PanicException, which is
            # not an Exception. Losing the reaction costs one decision; letting
            # it out of here costs the connection.
            logging.error('react(%s) failed: %r', kind, exc)
            answer = None
        elapsed = time.perf_counter() - started
        self.slowest = max(self.slowest, elapsed)

        # What a forfeited discard should give up, if it comes to that.
        if kind == 'tsumo' and event.get('actor') == self.seat:
            self.drawn = event.get('pai')
        elif kind == 'dahai' and event.get('actor') == self.seat:
            self.drawn = None

        if answer:
            self.reaction = json.loads(answer)
        if kind == 'end_game':
            self.finish(event)

    def respond(self, event):
        """Answer one `request_action`, echoing its id."""
        action, why = choose(self.reaction, event.get('possible_actions') or [],
                             self.seat if self.seat is not None else 0, self.drawn)
        if why:
            self.fallbacks += 1
            logging.warning('falling back to %s: %s', action.get('type'), why)
        self.reaction = None
        action['request_id'] = event.get('request_id')
        return action

    def finish(self, event):
        """A game ended: record where it placed and keep the log."""
        self.games += 1
        scores = event.get('scores') or []
        if self.seat is not None and len(scores) > self.seat:
            mine = scores[self.seat]
            # Placement by score, the seat's own rank among the four.
            place = 1 + sum(1 for s in scores if s > mine)
            self.results.append((place, mine))
            ranks = [p for p, _ in self.results]
            logging.info('game %d: %s, placed %d of 4 with %d; '
                         'average placement so far %.3f over %d',
                         self.games, scores, place, mine,
                         sum(ranks) / len(ranks), len(ranks))
        if self.log_dir and self.lines:
            os.makedirs(self.log_dir, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
            out = path.join(self.log_dir, f'{stamp}-seat{self.seat}.json')
            with open(out, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.lines) + '\n')
            logging.info('log written to %s', out)
        self.bot = None
        self.lines = []


async def play(url, token, session, games, ping_interval=20):
    """One connection, until the game budget is met or the socket closes."""
    import websockets

    headers = {'Authorization': f'Bearer {token}'}
    # websockets renamed this in 12.0 and kept the old name working for a
    # while; accept whichever this install has rather than pinning a version.
    try:
        connect = websockets.connect(url, additional_headers=headers,
                                     ping_interval=ping_interval)
    except TypeError:
        connect = websockets.connect(url, extra_headers=headers,
                                     ping_interval=ping_interval)

    async with connect as ws:
        logging.info('connected to %s', url)
        async for raw in ws:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logging.warning('not JSON, ignored: %.120s', line)
                continue
            kind = event.get('type')

            if kind == 'request_action':
                await ws.send(json.dumps(session.respond(event)))
            elif kind in MJAI_EVENTS:
                session.on_mjai(line, event)
                if kind == 'end_game' and games and session.games >= games:
                    logging.info('played %d games; closing', session.games)
                    await ws.close()
                    return True
            elif kind in VERDICT_EVENTS:
                logging.info('%s: %s', kind, json.dumps(event, ensure_ascii=False))
                session.verdicts.append(event)
            elif kind == 'error':
                logging.error('server: %s', event)
            elif kind not in CONTROL_EVENTS:
                logging.info('unknown message %.300s, ignored', line)
    return False


async def run(args, engine):
    session = Session(engine, args.log_dir)
    attempt = 0
    while True:
        try:
            done = await play(args.url, args.token, session, args.games)
            if done:
                return session
            logging.info('server closed the connection')
            attempt = 0
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:
            logging.error('connection failed: %r', exc)
            attempt += 1
        if args.once:
            return session
        # Backing off rather than hammering: a server that just refused us is
        # not helped by being asked again immediately, and a bot that spins on
        # a rejected token is how a token gets rate limited.
        delay = min(60, 2 ** min(attempt, 6))
        logging.info('reconnecting in %ds', delay)
        await asyncio.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--url', help='overrides --ranked')
    ap.add_argument('--ranked', action='store_true',
                    help='the ranked ladder instead of validation')
    ap.add_argument('--host', default='wss://game.riichi.dev')
    ap.add_argument('--state-file', default=DEFAULT_STATE_FILE,
                    required=DEFAULT_STATE_FILE is None,
                    help='the checkpoint to play'
                         + (' (default: the config\'s)' if DEFAULT_STATE_FILE else ''))
    ap.add_argument('--device', default='cpu',
                    help='cpu is enough: one decision at a time, and the grace '
                         'period is three seconds')
    ap.add_argument('--games', type=int, default=0, help='0 plays until stopped')
    ap.add_argument('--log-dir', help='write each game as mjai lines, for review')
    ap.add_argument('--once', action='store_true',
                    help='do not reconnect after a disconnection')
    ap.add_argument('--trust-checkpoint', action='store_true',
                    help='load a checkpoint that weights_only=True refuses; '
                         'only for a file you produced yourself')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    args.token = os.environ.get('RIICHI_BOT_TOKEN')
    if not args.token:
        raise SystemExit('RIICHI_BOT_TOKEN is not set; register a bot at '
                         'https://riichi.dev/bots and export its token')
    if not args.url:
        args.url = args.host + ('/ws/ranked' if args.ranked else '/ws/validate')

    engine, tag, steps, best = load_bot_engine(
        args.state_file, torch.device(args.device), args.trust_checkpoint)
    logging.info('playing %s from %s%s%s', tag, args.state_file,
                 f', step {steps:,}' if steps else '',
                 f', test play {best["avg_rank"]:.4f} / {best["avg_pt"]:+.3f}'
                 if best and 'avg_rank' in best else '')

    session = None
    try:
        session = asyncio.run(run(args, engine))
    except KeyboardInterrupt:
        pass
    if session and session.results:
        ranks = [p for p, _ in session.results]
        logging.info('%d games, average placement %.4f, %d fallbacks, '
                     'slowest decision %.0f ms',
                     len(ranks), sum(ranks) / len(ranks), session.fallbacks,
                     session.slowest * 1000)
        for verdict in session.verdicts:
            logging.info('%s', json.dumps(verdict, ensure_ascii=False))


if __name__ == '__main__':
    main()
