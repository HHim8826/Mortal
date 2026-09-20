"""What one step away from the argmax costs, measured by replaying the wall.

    python deviation_cost.py --policy logs/policy/policy-t0.05.pth \
        --seeds 25 --blocks 8 --out logs/deviation

Phase 3 rests on one number. `advantage_signal.py` asked whether decisions
that deviated from the argmax were followed by a smaller return, bucketed the
comparison by how sure the policy was, and reported the gap at 9.1 standard
errors. That standard error is computed as if decisions were independent; the
~30 decisions of a kyoku share one GRP delta, so they are not. And the
bucketing is not the same thing as holding the state fixed: sampling deviates
exactly where the policy is unsure, `V` differs systematically there, and
subtracting it pulls that difference into the quantity being compared.

The repair is not a better standard error. It is to stop asking the data what
a deviation was followed by and start asking the game what a deviation causes.
A wall is determined by its seed, so the same hand can be dealt twice: once
with the policy taking its argmax throughout, and once with the argmax
replaced at exactly one decision by the action sampling would have drawn.
Everything before that decision is identical, the opponents are deterministic,
and what follows differs by that one deviation and nothing else. No baseline,
no buckets, no independence assumption -- the pairing `evaluate.py` already
uses on whole games, moved down to a single decision.

Which decision is not chosen evenly. A deviation happens where the policy
leaves mass off its own best action, so the target is drawn with probability
proportional to `1 - p(argmax)` and the replacement from the rest of the
distribution. What comes out is the cost of a deviation where deviations
really occur, and multiplying by how many a game sees is comparable with phase
2's end-to-end measurement of sampling -- a known answer to check against.

What this does not measure: real sampling deviates repeatedly and the
deviations interact, so the total is not the per-deviation cost times the
rate. This is the cost of the first step off an argmax line, which is the
quantity a policy gradient sees one decision at a time.
"""
import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from os import path

import numpy as np
import torch

import prelude                                          # noqa: F401
import rollout as ro
from config import config
from engine import MortalEngine
from model import Brain, GRP, PolicyHead
from reward_calculator import RewardCalculator

CHALLENGER = 'trainee'
# The placement points every strength number in this project is quoted in --
# `evaluate.py` owns them, and phase 2's -1.13 +- 1.53 for sampling end to end
# is on this scale. The kyoku's GRP delta is a different one, `[env] pts`, and
# the two must not be added up or compared without saying which is which.
from evaluate import PTS

def load_policy(file, device):
    """The policy under test, as the trunk and head an engine plays."""
    state = torch.load(file, weights_only=True, map_location='cpu')
    if 'policy' not in state:
        raise SystemExit(f'{file} has no policy head; train one with train_policy.py')
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    brain = Brain(version=version, conv_channels=cfg['resnet']['conv_channels'],
                  num_blocks=cfg['resnet']['num_blocks']).eval()
    brain.load_state_dict(state['mortal'])
    head = PolicyHead(version=version).eval()
    head.load_state_dict(state['policy'])
    return (brain.to(device).requires_grad_(False),
            head.to(device).requires_grad_(False), version,
            state.get('play_temperature'))

def cheap_of(obs):
    """A fingerprint to look a state up by, cheap enough to take on every row.

    A v4 observation is 1012x34 floats, and hashing all of it for every
    decision of every game would cost more than the forward pass it rides
    along with. Every 97th number is enough to miss with, and `full_of`
    settles the ones that hit.
    """
    flat = np.asarray(obs).reshape(-1)[::97]
    return hashlib.blake2b(np.ascontiguousarray(flat).tobytes(), digest_size=8).digest()

def full_of(obs, mask):
    """The state itself, named. What a fingerprint match is confirmed against."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(np.asarray(obs)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(mask, dtype=bool)).tobytes())
    return h.digest()

class Forcer:
    """Plays what the engine underneath plays, except at the states it is armed with.

    The arena calls one engine for every game it is carrying at once, so a
    decision cannot be named by counting calls: which batch a game lands in
    depends on what else is still running, and after a fork that differs. The
    state itself is the name that survives it.
    """

    def __init__(self, base):
        self.base = base
        self.armed = defaultdict(list)      # fingerprint -> [(state, action, tag)]
        self.fired = {}
        self.collisions = 0

    def arm(self, cheap, full, action, tag):
        self.armed[cheap].append((full, int(action), tag))

    def __getattr__(self, name):
        # Everything the arena asks of an engine that this does not override.
        return getattr(self.base, name)

    def react_batch(self, obs, masks, invisible_obs):
        actions, q_out, mask_out, is_greedy = self.base.react_batch(obs, masks, invisible_obs)
        if not self.armed:
            return actions, q_out, mask_out, is_greedy
        for i, one in enumerate(obs):
            candidates = self.armed.get(cheap_of(one))
            if not candidates:
                continue
            full = full_of(one, masks[i])
            for state, action, tag in candidates:
                if state != full:
                    self.collisions += 1
                    continue
                if tag in self.fired:
                    # The same state twice is the pairing's assumption
                    # breaking, not a second chance to force the action.
                    raise SystemExit(f'the state armed for {tag} came round twice')
                self.fired[tag] = dict(was=int(actions[i]), forced=action)
                actions[i] = action
        return actions, q_out, mask_out, is_greedy

def play(challenger, champion, seed_start, seed_count, log_dir):
    """One block of seeds, four hanchans each, into a directory of its own."""
    from libriichi.arena import OneVsThree
    if path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    env = OneVsThree(disable_progress_bar=True, log_dir=log_dir)
    env.py_vs_py(challenger=challenger, champion=champion,
                 seed_start=seed_start, seed_count=seed_count)
    return sorted(f for f in os.listdir(log_dir) if f.endswith('.json.gz'))

def read(log_dir, name):
    with gzip.open(path.join(log_dir, name), 'rt', encoding='utf-8') as f:
        return f.read()

def seat_of(log):
    return json.loads(log.split('\n', 1)[0])['names'].index(CHALLENGER)

def bare(log):
    """A log as the moves that were made, with everything else dropped.

    A `meta` is about the run, not the play: how long the forward took, how
    many games shared the batch, and the logits themselves, which move in the
    last bits when the batch a game rides in changes size. Measured over 100
    replayed hanchans with half precision on, that perturbation reached
    3.3e-2 and left the play identical in 49 of 50 -- small, and not always
    small enough. What a witness has to answer is whether the moves changed,
    so the numbers beside them come out.
    """
    out = []
    for line in log.splitlines():
        if not line:
            continue
        event = json.loads(line)
        event.pop('meta', None)
        out.append(json.dumps(event, sort_keys=True))
    return out

def first_divergence(one, other):
    """The event index where two logs stop being the same game, or None.

    Where a fork was armed this should be the forced decision itself. Anything
    earlier means the line was already moving before anything was done to it.
    """
    a, b = bare(one), bare(other)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))

def same_play(one, other):
    return first_divergence(one, other) is None

def decoded(loader, log, seat):
    games = loader.load_logs([log])[0]
    return next(g for g in games if g.take_player_id() == seat)

def pick_target(log, seat, game, rng):
    """One decision to fork, drawn where deviations actually happen.

    None for a hanchan the policy was never given a real choice in.
    """
    obs, masks = game.take_obs(), game.take_masks()
    actions, at_kyoku = game.take_actions(), game.take_at_kyoku()
    metas = ro.aligned_metas(actions, masks, log, seat)

    rows = []
    for i, meta in enumerate(metas):
        if meta is None:
            continue
        ids = np.flatnonzero(np.asarray(masks[i], dtype=bool))
        if len(ids) < 2:
            continue
        logits = np.asarray(meta['q_values'], dtype=np.float64)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        best = int(p.argmax())
        off = 1. - p[best]
        if off <= 1e-9:
            continue
        rows.append((i, ids, p, best, off))
    if not rows:
        return None

    weights = np.array([r[4] for r in rows])
    i, ids, p, best, off = rows[rng.choice(len(rows), p=weights / weights.sum())]
    # The replacement, drawn from the policy's own distribution with its best
    # action taken out: exactly the draw that produces a deviation.
    rest = p.copy()
    rest[best] = 0.
    rest /= rest.sum()
    alt = int(rng.choice(len(ids), p=rest))
    return dict(
        index=int(i), kyoku=int(at_kyoku[i]),
        argmax=int(ids[best]), forced=int(ids[alt]),
        p_argmax=float(p[best]), p_forced=float(p[alt]), p_deviate=float(off),
        deviations_expected=float(sum(r[4] for r in rows)), decisions=len(rows),
        cheap=cheap_of(obs[i]), full=full_of(obs[i], masks[i]),
    )

def outcomes(seat, game, reward_calc):
    """What the hanchan paid this seat, per kyoku and at the end."""
    grp = game.take_grp()
    deltas = reward_calc.calc_delta_pt(seat, grp.take_feature(), grp.take_rank_by_player())
    final = np.asarray(grp.take_final_scores())
    rank = int((final > final[seat]).sum())
    return dict(
        kyoku_delta=np.asarray(deltas, dtype=np.float64),
        score=float(final[seat]),
        rank=rank,
        pt=float(PTS[rank]),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True, help='the policy under test')
    ap.add_argument('--out', default='logs/deviation')
    ap.add_argument('--seeds', type=int, default=25,
                    help='seeds per block, four hanchans each')
    ap.add_argument('--blocks', type=int, default=1)
    ap.add_argument('--seed-start', type=int, default=900000)
    ap.add_argument('--key', type=int, default=0xD1FF,
                    help='the wall key, fixed so a run can be repeated')
    ap.add_argument('--rng', type=int, default=0, help='which decision gets forked')
    ap.add_argument('--device', default=None)
    ap.add_argument('--per-hanchan', type=int, default=1,
                    help='targets armed per hanchan; 0 replays with nothing armed, '
                         'which is the determinism check on its own')
    ap.add_argument('--witness-every', type=int, default=10,
                    help='leave one hanchan in this many unforked, to watch how often '
                         'forking one game reaches another through the batch they share')
    ap.add_argument('--contamination', type=float, default=0.10,
                    help='stop if more than this share of witnesses moved')
    ap.add_argument('--amp', action='store_true',
                    help='play in half precision, as the workers do. Off by default '
                         'here: see the note on Forcer')
    ap.add_argument('--keep-logs', action='store_true',
                    help='leave each block behind instead of overwriting it')
    args = ap.parse_args()

    device = torch.device(args.device or config['control']['device'])
    brain, head, version, temperature = load_policy(args.policy, device)
    logging.info(f'{args.policy}: v{version}, play temperature {temperature}')

    def engine(name):
        # Argmax, the guard off, amp on: the settings the policy-gradient
        # workers played under, minus the sampling this replaces with one
        # chosen deviation.
        return MortalEngine(brain, head, is_oracle=False, version=version, device=device,
                            enable_amp=args.amp, enable_rule_based_agari_guard=False,
                            name=name, boltzmann_epsilon=0.)

    champion = engine('champion')
    from libriichi.dataset import GameplayLoader
    loader = GameplayLoader(version=version, oracle=False)
    grp = GRP(**config['grp']['network'])
    grp.load_state_dict(torch.load(config['grp']['state_file'], weights_only=True,
                                   map_location='cpu')['model'])
    reward_calc = RewardCalculator(grp, config['env']['pts'])

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.rng)
    rows, identical, changed, witnesses = [], 0, 0, []

    for block in range(args.blocks):
        first = args.seed_start + block * args.seeds
        tag = f'block{block}' if args.keep_logs else 'block'
        base_dir = path.join(args.out, f'{tag}-base')
        fork_dir = path.join(args.out, f'{tag}-fork')

        names = play(engine(CHALLENGER), champion, (first, args.key), args.seeds, base_dir)
        base = {}
        for name in names:
            log = read(base_dir, name)
            seat = seat_of(log)
            base[name] = (log, seat, decoded(loader, log, seat))

        forcer = Forcer(engine(CHALLENGER))
        targets = {}
        for n, (name, (log, seat, game)) in enumerate(base.items()):
            if args.per_hanchan < 1:
                continue
            if args.witness_every and n % args.witness_every == 0:
                continue
            target = pick_target(log, seat, game, rng)
            if target is None:
                continue
            targets[name] = target
            forcer.arm(target['cheap'], target['full'], target['forced'], name)

        play(forcer, champion, (first, args.key), args.seeds, fork_dir)
        missed = [n for n in targets if n not in forcer.fired]
        for name in missed:
            # Where did it go wrong? A line that had already left the baseline
            # before its own target is the batch reaching between games; one
            # that never left is something else, and the two want different
            # fixes.
            where = first_divergence(base[name][0], read(fork_dir, name))
            logging.warning(f'{name}: armed state never came round, first divergence at '
                            f'event {where} (target was decision {targets[name]["index"]})')
            del targets[name]

        for name in names:
            log_b, seat, game_b = base[name]
            log_f = read(fork_dir, name)
            if name not in targets:
                # Nothing was forked here, so it is a witness: if it moved,
                # the two arms are not comparable and nor is anything else in
                # the block.
                if same_play(log_b, log_f):
                    identical += 1
                else:
                    changed += 1
                    witnesses.append(name)
                continue
            forked_at = first_divergence(log_b, log_f)
            if forked_at is None:
                # The action was replaced and the game came out identical --
                # a deviation that changed nothing is a real outcome, not an
                # error, but it should be visible rather than assumed.
                logging.debug(f'{name}: forced action left the log unchanged')
            out_b = outcomes(seat, game_b, reward_calc)
            out_f = outcomes(seat, decoded(loader, log_f, seat), reward_calc)
            t = targets[name]
            k = t['kyoku']
            rows.append(dict(
                block=block, log=name,
                **{x: t[x] for x in ('index', 'kyoku', 'argmax', 'forced', 'p_argmax',
                                     'p_forced', 'p_deviate', 'deviations_expected',
                                     'decisions')},
                was=forcer.fired[name]['was'], forked_at=forked_at,
                kyoku_base=float(out_b['kyoku_delta'][k]),
                kyoku_fork=(float(out_f['kyoku_delta'][k])
                            if k < len(out_f['kyoku_delta']) else float('nan')),
                score_base=out_b['score'], score_fork=out_f['score'],
                pt_base=out_b['pt'], pt_fork=out_f['pt'],
                rank_base=out_b['rank'], rank_fork=out_f['rank'],
            ))
        logging.info(f'block {block}: {len(targets)} forked, {identical:,} untouched '
                     f'hanchans identical, {changed:,} not, {forcer.collisions} '
                     'fingerprint collisions')
        if missed:
            logging.warning(f'block {block}: dropped {len(missed)} of '
                            f'{len(missed) + len(targets)} targets that never came round')
        seen = identical + changed
        if seen and changed / seen > args.contamination:
            raise SystemExit(
                f'block {block}: {changed} of {seen} hanchans with nothing forked in them '
                f'came out playing differently. Forking one game is reaching the others '
                'often enough that the drop rate is no longer incidental: play without '
                '--amp, or put fewer games in one arena.')

    report(rows, identical, changed, args)

def report(rows, identical, changed, args):
    out = path.join(args.out, 'deviations.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(args), identical=identical, changed=changed, rows=rows),
                  f, indent=1)

    print()
    print(f'untouched hanchans: {identical:,} identical, {changed:,} not')
    if not rows:
        print('nothing was forked, so there is nothing else to report')
        print(f'\nwritten to {out}')
        return

    kyoku = np.array([r['kyoku_fork'] - r['kyoku_base'] for r in rows])
    pts = np.array([r['pt_fork'] - r['pt_base'] for r in rows])
    score = np.array([r['score_fork'] - r['score_base'] for r in rows])
    rank = np.array([float(r['rank_fork'] - r['rank_base']) for r in rows])
    per_game = np.array([r['deviations_expected'] for r in rows])
    decisions = np.array([r['decisions'] for r in rows])

    def line(name, x):
        n = len(x)
        se = x.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
        print(f'{name:>28}  {x.mean():+10.4f} +- {se:8.4f}  ({x.mean() / se:+5.1f} se, n={n:,})')

    print(f'{len(rows):,} forced deviations, one per hanchan, each against the same wall '
          'played by the same policy taking its argmax')
    print()
    print('what the one deviation changed')
    line("the kyoku's GRP delta", kyoku[np.isfinite(kyoku)])
    line('the hanchan, in pt', pts)
    line('the hanchan, in placement', rank)
    line('the hanchan, in score', score)
    print()
    print(f'a hanchan has {decisions.mean():.0f} decisions with a choice in it, of which '
          f'{per_game.mean():.2f} would deviate')
    total = per_game.mean() * pts.mean()
    se = per_game.mean() * pts.std(ddof=1) / np.sqrt(len(pts))
    print(f'so sampling end to end, if deviations did not interact: '
          f'{total:+.2f} +- {se:.2f} pt')
    print('phase 2 measured that at -1.13 +- 1.53 pt. They are different experiments -- '
          'that one sampled every decision, this one forces a single deviation onto an '
          'argmax line -- so they should agree in sign and order, not exactly')
    print(f'\nwritten to {out}')

if __name__ == '__main__':
    main()
