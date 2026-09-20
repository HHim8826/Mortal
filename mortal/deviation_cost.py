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
import multiprocessing
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
from dataloader import digest64
from model import Brain, GRP, PolicyHead
from reward_calculator import RewardCalculator

CHALLENGER = 'trainee'
# The placement points every strength number in this project is quoted in --
# `evaluate.py` owns them, and phase 2's -1.13 +- 1.53 for sampling end to end
# is on this scale. The kyoku's GRP delta is a different one, `[env] pts`, and
# the two must not be added up or compared without saying which is which.
from evaluate import PTS

def load_policy(file):
    """The policy under test, as the trunk and head an engine plays.

    Left on the CPU. Moving it to the GPU initialises CUDA, and the worker
    pool is forked before that happens on purpose -- see `start_workers`.
    """
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
    return (brain.requires_grad_(False), head.requires_grad_(False), version,
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

# Decoding a log is most of this program. The arena plays 2,000 hanchans a
# block in about ten minutes on eight of twelve cores; reading them back --
# three decodes a hanchan, a GRP forward for each arm, and a JSON parse of
# every event twice over -- took nearly half an hour on one. It is per-log
# work with nothing shared, so it goes to a pool. These two are what the pool
# runs, and the loader and the reward calculator they use are built in the
# parent before the fork, so every worker inherits one rather than paying to
# build its own.
_LOADER = None
_REWARD = None

def start_workers(jobs):
    """The pool, forked before this process has anything worth inheriting.

    Twice before, not once. A child of a process with a live CUDA context
    must never touch CUDA, and the simplest way to be sure is for the context
    not to exist yet -- so this is called before the weights move to the GPU.
    And `fork` from a process that already has threads can deadlock the child
    on a lock no surviving thread will release, which torch gives it as soon
    as it does any real work, so this is called before that too. The children
    build their own decoder on first use rather than inheriting one.
    """
    if jobs < 2:
        return None
    return multiprocessing.get_context('fork').Pool(jobs)

def _state(version):
    """This worker's decoder and GRP, built once and kept."""
    global _LOADER, _REWARD
    if _LOADER is None:
        # One thread each. Ten workers all reaching for twelve cores is
        # slower than ten workers taking one apiece.
        torch.set_num_threads(1)
        from libriichi.dataset import GameplayLoader
        _LOADER = GameplayLoader(version=version, oracle=False)
        grp = GRP(**config['grp']['network'])
        grp.load_state_dict(torch.load(config['grp']['state_file'],
                                       weights_only=True, map_location='cpu')['model'])
        _REWARD = RewardCalculator(grp, config['env']['pts'])
    return _LOADER, _REWARD

def _pick_job(job):
    """One log, read and searched for a decision worth forking."""
    version, base_dir, name, seed, pick, rule = job
    loader, _ = _state(version)
    log = read(base_dir, name)
    seat = seat_of(log)
    target = pick_target(log, seat, decoded(loader, log, seat),
                         np.random.default_rng(seed), pick, rule)
    if target is not None:
        target['seat'] = seat
    return name, target

def _measure_job(job):
    """One hanchan's two arms, compared. A target of None is a witness."""
    version, base_dir, fork_dir, name, target = job
    one, other = read(base_dir, name), read(fork_dir, name)
    if target is None:
        return name, dict(witness=same_play(one, other))
    loader, reward = _state(version)
    seat, k = target['seat'], target['kyoku']
    base = outcomes(seat, decoded(loader, one, seat), reward)
    fork = outcomes(seat, decoded(loader, other, seat), reward)
    return name, dict(
        forked_at=first_divergence(one, other),
        kyoku_base=float(base['kyoku_delta'][k]),
        kyoku_fork=(float(fork['kyoku_delta'][k])
                    if k < len(fork['kyoku_delta']) else float('nan')),
        score_base=base['score'], score_fork=fork['score'],
        scores_base=base['scores'], scores_fork=fork['scores'],
        pt_base=base['pt'], pt_fork=fork['pt'],
        rank_base=base['rank'], rank_fork=fork['rank'],
    )

def spread(pool, fn, jobs):
    """`fn` over `jobs`, in the pool if there is one."""
    if pool is None:
        return [fn(job) for job in jobs]
    return pool.map(fn, jobs, chunksize=8)

def assign_rules(names, rules, witness_every, rng):
    """Which rule each hanchan gets, and which are left alone as witnesses.

    One wall is played four times with the challenger in each seat, and the
    arena names them a, b, c, d in that order. Handing the rules out by
    position therefore gave every rule its own starting seat for the whole
    run -- argmax always seat 0, worst always seat 3 -- and no number of
    blocks would shake that loose. Seats get different hands, a different
    turn order and a different side of a tie, so a difference between the
    rules would have been partly a difference between seats.

    Each wall draws its own permutation instead, which balances the rules
    across the seats by construction. Witnesses are drawn at random for the
    same reason: every tenth file is seat 0 or seat 2 and never the other two.

    The rule counter runs across the walls rather than restarting at each
    one. A wall holds four hanchans, so restarting it meant a fifth rule was
    never reached at all: asking for all five ran a whole evaluation that
    measured four of them and said nothing about the fifth.
    """
    walls = defaultdict(list)
    for name in names:
        walls[name.rsplit('_', 1)[0]].append(name)
    assigned, witnesses = {}, set()
    slot = 0
    for wall in sorted(walls):
        members = sorted(walls[wall])
        for which in rng.permutation(len(members)):
            name = members[which]
            if witness_every and rng.random() < 1 / witness_every:
                witnesses.add(name)
            else:
                assigned[name] = rules[slot % len(rules)]
            slot += 1
    return assigned, witnesses

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

# Which action to put in the argmax's place. `sampled` is what sampling
# would really have drawn, and answers what deviating costs. The rest are
# places in the policy's own ordering, and answer a different question: is
# that ordering right? `argmax` replaces the best action with itself, so it
# must come out at exactly zero -- the null this whole apparatus is checked
# against.
RULES = ('sampled', 'argmax', 'rank2', 'median', 'worst')

def pick_target(log, seat, game, rng, pick='deviation', rule='sampled'):
    """One decision to fork, and what to play there instead.

    `pick` is how the decision is drawn. `deviation` weights it by
    `1 - p(argmax)`, so the measurement lands where deviations really happen
    and says what sampling costs. `uniform` draws evenly over every decision
    with a choice in it, which is what comparing rules needs: the rules are
    only comparable if they are answered on the same states, and weighting by
    how unsure the policy is would hand each rule a different population.

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
        if pick == 'deviation' and off <= 1e-9:
            continue
        rows.append((i, ids, p, best, off))
    if not rows:
        return None

    weights = (np.array([r[4] for r in rows]) if pick == 'deviation'
               else np.ones(len(rows)))
    if not weights.sum():
        return None
    i, ids, p, best, off = rows[rng.choice(len(rows), p=weights / weights.sum())]
    order = np.argsort(-p, kind='stable')
    match rule:
        case 'sampled':
            # What sampling would really have drawn: the policy's own
            # distribution with its best action taken out.
            rest = p.copy()
            rest[best] = 0.
            if not rest.sum():
                return None
            alt = int(rng.choice(len(ids), p=rest / rest.sum()))
        case 'argmax':
            alt = int(order[0])
        case 'rank2':
            alt = int(order[1])
        case 'median':
            alt = int(order[len(order) // 2])
        case 'worst':
            alt = int(order[-1])
        case _:
            raise ValueError(f'unknown rule {rule!r}')
    return dict(
        index=int(i), kyoku=int(at_kyoku[i]), rule=rule,
        argmax=int(ids[best]), forced=int(ids[alt]),
        # Where the forced action sits in the policy's ordering, and how many
        # it was chosen from. `median` and `worst` are the same action in a
        # two-way choice, and this is what says so afterwards.
        forced_rank=int(np.flatnonzero(order == alt)[0]), legal=len(ids),
        p_argmax=float(p[best]), p_forced=float(p[alt]), p_deviate=float(off),
        deviations_expected=float(sum(r[4] for r in rows)), decisions=len(rows),
        cheap=cheap_of(obs[i]), full=full_of(obs[i], masks[i]),
    )

def outcomes(seat, game, reward_calc):
    """What the hanchan paid this seat, per kyoku and at the end.

    The placement comes from the engine, not from counting who scored more.
    Two seats can finish level and the engine still ranks them, by where they
    started; counting strictly-greater scores calls them both the higher
    place. That is a whole 45 pt on the Tenhou scale, it lands on one arm of
    a pair and not the other, and it does not cancel in the difference. It is
    also the ranking `calc_delta_pt` is already being handed on the line
    above, so taking it from anywhere else was two answers to one question.
    """
    grp = game.take_grp()
    rank_by_player = grp.take_rank_by_player()
    deltas = reward_calc.calc_delta_pt(seat, grp.take_feature(), rank_by_player)
    final = np.asarray(grp.take_final_scores())
    rank = int(rank_by_player[seat])
    return dict(
        kyoku_delta=np.asarray(deltas, dtype=np.float64),
        score=float(final[seat]),
        scores=[float(x) for x in final],
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
    ap.add_argument('--pick', default='deviation', choices=('deviation', 'uniform'),
                    help='how the decision to fork is drawn. deviation weights it by '
                         '1 - p(argmax) and measures what sampling costs; uniform draws '
                         'evenly and is what comparing --force rules needs')
    ap.add_argument('--force', default='sampled',
                    help='comma separated rules from ' + ','.join(RULES) + ', cycled '
                         'over the hanchans of a block so every rule meets the same '
                         'states. argmax replaces the best action with itself and must '
                         'come out at exactly zero')
    ap.add_argument('--jobs', type=int, default=0,
                    help='processes to read the logs back with. 0 picks two fewer '
                         'than the machine has, 1 keeps it in this process')
    ap.add_argument('--keep-logs', action='store_true',
                    help='leave each block behind instead of overwriting it')
    args = ap.parse_args()

    # First, before this process has a CUDA context or a thread pool for a
    # child to inherit and deadlock on.
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 2)
    pool = start_workers(jobs)

    device = torch.device(args.device or config['control']['device'])
    brain, head, version, temperature = load_policy(args.policy)
    logging.info(f'{args.policy}: v{version}, play temperature {temperature}, '
                 f'logs read back across {jobs} processes')
    brain, head = brain.to(device), head.to(device)

    def engine(name):
        # Argmax, the guard off, amp on: the settings the policy-gradient
        # workers played under, minus the sampling this replaces with one
        # chosen deviation.
        return MortalEngine(brain, head, is_oracle=False, version=version, device=device,
                            enable_amp=args.amp, enable_rule_based_agari_guard=False,
                            name=name, boltzmann_epsilon=0.)

    champion = engine('champion')

    os.makedirs(args.out, exist_ok=True)
    rules = args.force.split(',')
    for rule in rules:
        if rule not in RULES:
            raise SystemExit(f'unknown --force rule {rule!r}: expected {", ".join(RULES)}')
    logging.info(f'picking decisions {args.pick}ly, forcing {"/".join(rules)}')
    rows, identical, changed, witnesses, mismatched = [], 0, 0, [], []
    rejected = 0

    for block in range(args.blocks):
        first = args.seed_start + block * args.seeds
        tag = f'block{block}' if args.keep_logs else 'block'
        base_dir = path.join(args.out, f'{tag}-base')
        fork_dir = path.join(args.out, f'{tag}-fork')

        names = play(engine(CHALLENGER), champion, (first, args.key), args.seeds, base_dir)

        # One log at a time, and nothing decoded kept afterwards. A v4
        # observation is 1012x34 floats and a hanchan holds a few hundred of
        # them, so a decoded game is tens of megabytes; holding a block of a
        # thousand was 40 GB and the run was killed for it. The target is a
        # hash and a handful of numbers, and the logs are still on disk.
        forcer = Forcer(engine(CHALLENGER))
        assigned = {}
        if args.per_hanchan >= 1:
            assigned, _ = assign_rules(names, rules, args.witness_every,
                                       np.random.default_rng([args.rng, block, 1]))
        # `digest64` is signed, for the tensor it usually ends up in; a seed
        # has to be non-negative.
        wanted = [(version, base_dir, name,
                   [args.rng, block, digest64(name) % (1 << 63)], args.pick, rule)
                  for name, rule in sorted(assigned.items())]
        targets = {}
        for name, target in spread(pool, _pick_job, wanted):
            if target is None:
                continue
            targets[name] = target
            forcer.arm(target['cheap'], target['full'], target['forced'], name)
        # Armed is not the same as interfered with, and the difference is
        # exactly the evidence this run is looking for. A target that fired
        # and was dropped afterwards did have its action replaced, so its two
        # arms differ by design and counting it as a hanchan nothing was done
        # to would report the intervention as contamination. A target that
        # never fired had nothing replaced -- it is an untouched replay, and
        # if it came out different that is another game's fork reaching it
        # through the batch they share, which is the one thing the witnesses
        # exist to catch. `forcer.fired` is what tells them apart, so it is
        # read after the fork has been played.

        play(forcer, champion, (first, args.key), args.seeds, fork_dir)
        mismatched_before = len(mismatched)
        # The alignment between a log's events and the instances decoded from
        # it walks a cursor, and a declined call leaves no event to move it:
        # a pass and a later pon can carry the same legal-action mask, and the
        # pass then takes the pon's logits. A target picked from the wrong
        # distribution names the wrong argmax, and firing only proves the
        # state was found, not that the numbers attached to it belong to it.
        # The baseline takes its argmax, so the two must agree.
        for name in list(targets):
            fired = forcer.fired.get(name)
            if fired is not None and fired['was'] != targets[name]['argmax']:
                logging.warning(
                    f'{name}: the metadata says the argmax here is '
                    f'{targets[name]["argmax"]} and the policy played {fired["was"]}, '
                    'so this decision was paired with the logits of another one')
                mismatched.append(name)
                del targets[name]

        interfered = set(forcer.fired)
        missed = [n for n in targets if n not in forcer.fired]
        for name in missed:
            # Where did it go wrong? A line that had already left the baseline
            # before its own target is the batch reaching between games; one
            # that never left is something else, and the two want different
            # fixes.
            where = first_divergence(read(base_dir, name), read(fork_dir, name))
            logging.warning(f'{name}: armed state never came round, first divergence at '
                            f'event {where} (target was decision {targets[name]["index"]})')
            del targets[name]

        for name, got in spread(pool, _measure_job,
                                [(version, base_dir, fork_dir, name,
                                  targets.get(name)) for name in names]):
            if 'witness' in got:
                if name in interfered:
                    # Forked, then the measurement was thrown away. Not a
                    # witness either way.
                    rejected += 1
                    continue
                # Nothing was forked here, so it is a witness: if it moved,
                # the two arms are not comparable and nor is anything else in
                # the block.
                if got['witness']:
                    identical += 1
                else:
                    changed += 1
                    witnesses.append(name)
                continue
            if got['forked_at'] is None:
                # The action was replaced and the game came out identical --
                # a deviation that changed nothing is a real outcome, not an
                # error, but it should be visible rather than assumed.
                logging.debug(f'{name}: forced action left the log unchanged')
            t = targets[name]
            rows.append(dict(
                block=block, log=name,
                **{x: t[x] for x in ('index', 'kyoku', 'seat', 'rule', 'argmax',
                                     'forced', 'forced_rank', 'legal', 'p_argmax',
                                     'p_forced', 'p_deviate', 'deviations_expected',
                                     'decisions')},
                was=forcer.fired[name]['was'], **got))
        logging.info(f'block {block}: {len(targets)} forked, {identical:,} untouched '
                     f'hanchans identical, {changed:,} not, {rejected} forked but '
                     f'dropped, {forcer.collisions} fingerprint collisions')
        save(rows, identical, changed, len(mismatched), rejected, args)
        if new_mismatches := len(mismatched) - mismatched_before:
            logging.warning(f'block {block}: dropped {new_mismatches} targets whose '
                            f'logits belong to another decision ({len(mismatched)} so far)')
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

    report(rows, identical, changed, len(mismatched), rejected, args)

def save(rows, identical, changed, mismatched, rejected, args):
    """Everything measured so far, rewritten after every block.

    A block is a quarter of an hour of play and the blocks after it can still
    fail; what has already been measured should survive that.
    """
    out = path.join(args.out, 'deviations.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(args), identical=identical, changed=changed,
                       mismatched=mismatched, rejected=rejected, rows=rows), f, indent=1)
    return out

def report(rows, identical, changed, mismatched, rejected, args):
    out = save(rows, identical, changed, mismatched, rejected, args)
    print()
    print(f'untouched hanchans: {identical:,} identical, {changed:,} not'
          + (f'; {rejected} more were forked and then dropped, which is not the same '
             'thing and is not counted here' if rejected else ''))
    if mismatched:
        print(f'targets dropped for borrowed logits: {mismatched:,}')
    if not rows:
        print('nothing was forked, so there is nothing else to report')
        print(f'\nwritten to {out}')
        return

    by_rule = {}
    for row in rows:
        by_rule.setdefault(row.get('rule', 'sampled'), []).append(row)
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

    if len(by_rule) > 1:
        print()
        print("the kyoku's GRP delta, by which action was put in the argmax's place")
        print(f'{"":>10}  {"decisions":>9}  {"rank":>5}  {"p(forced)":>9}  {"effect":>20}')
        for rule in RULES:
            here = by_rule.get(rule)
            if not here:
                continue
            x = np.array([r['kyoku_fork'] - r['kyoku_base'] for r in here], float)
            x = x[np.isfinite(x)]
            se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')
            where = np.mean([r['forced_rank'] for r in here])
            chance = np.mean([r['p_forced'] for r in here])
            print(f'{rule:>10}  {len(here):>9,}  {where:>5.1f}  {chance:>9.3f}  '
                  f'{x.mean():>+9.4f} +- {se:.4f}')
        if 'argmax' in by_rule:
            print('argmax replaces the best action with itself, so anything but a row of '
                  'zeros there means the apparatus is measuring something it should not')

    print()
    print('what the one deviation changed, over every rule together')
    line("the kyoku's GRP delta", kyoku[np.isfinite(kyoku)])
    line('the hanchan, in pt', pts)
    line('the hanchan, in placement', rank)
    line('the hanchan, in score', score)
    print()
    print(f'a hanchan has {decisions.mean():.0f} decisions with a choice in it, of which '
          f'{per_game.mean():.2f} would deviate')
    # Only one configuration estimates what sampling costs: the decision has
    # to be drawn in proportion to how likely a deviation was there, and the
    # action has to be the one sampling would have drawn. Under `--pick
    # uniform` the target came from a different distribution, and under any
    # other rule the action did; multiplying by the deviation count then
    # answers no question at all, and on synthetic rows it comes out the
    # wrong sign. So it is not printed.
    if args.pick == 'deviation' and set(by_rule) == {'sampled'}:
        # Per hanchan, not per average hanchan: each game's own deviation
        # count times its own measured cost. Multiplying the two averages
        # would assume a game's deviation rate says nothing about what its
        # deviations cost, which nothing here establishes.
        whole = per_game * pts
        print(f'so sampling end to end, if deviations did not interact: '
              f'{whole.mean():+.2f} +- {whole.std(ddof=1) / np.sqrt(len(whole)):.2f} pt')
        print('phase 2 measured that at -1.13 +- 1.53 pt. They are different experiments '
              '-- that one sampled every decision, this one forces a single deviation '
              'onto an argmax line -- so they should agree in sign and order, not exactly')
    else:
        print('what that costs end to end is not estimated here: it needs the decision '
              'drawn by how likely a deviation was and the action drawn as sampling '
              'would have drawn it, which is --pick deviation --force sampled')
    print(f'\nwritten to {out}')

if __name__ == '__main__':
    main()
