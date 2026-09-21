"""The states the paired replays measured, recovered, and what the trunk makes of them.

`deviation_cost.py` produces labels: for one decision, what taking a
different action there did to the kyoku's return, measured against the same
wall. It does not keep the state those labels belong to. Without
`--keep-logs` each block overwrites the last, so after a ten-block run only
the final thousand hanchans are still on disk and the other nine tenths of
the inputs are gone.

They are not lost, though, because nothing about the baseline arm was
random: the wall comes from its seed, the opponents are deterministic, and
the challenger takes its argmax. Replaying that one arm reproduces the same
logs, and each row already records which decision of which hanchan it was.
Only the baseline is replayed here -- no fork, no second GRP pass -- so this
costs about half of what the original run did.

Replaying is a claim that has to be checked rather than assumed, so the pick
is run again with the same seed and the recovered target is compared against
the stored row field by field. A log that came back even slightly different
lands on a different decision and is thrown out by name.

What comes out is what a probe needs: the 1024 trunk features of each
measured state, the mask, the two actions compared, and the causal
advantage between them.

    python deviation_features.py --results /root/eval/deviation/sweep/deviations.json \
        --policy logs/policy/policy-t0.05.pth
"""
import argparse
import json
import logging
import os
from collections import defaultdict
from os import path

import numpy as np
import torch

import prelude                                          # noqa: F401
from config import config
from deviation_cost import (CHALLENGER, _state, decoded, load_policy, pick_target, play,
                            read, seat_of, spread, start_workers)
from dataloader import digest64
from engine import MortalEngine

# What must come back identical for a replayed log to count as the same log.
# The index alone would be weak: the same decision number in a different game
# is still a number. These pin the decision, the choice on offer, and what
# the policy thought of it.
SAME = ('index', 'kyoku', 'argmax', 'forced', 'forced_rank', 'legal')
CLOSE = ('p_argmax', 'p_forced')


def _feature_job(job):
    """One log, replayed and searched for the decision a row already names."""
    version, base_dir, name, seed, pick, rule, only = job
    loader, _ = _state(version)
    log = read(base_dir, name)
    seat = seat_of(log)
    target = pick_target(log, seat, decoded(loader, log, seat),
                         np.random.default_rng(seed), pick, rule, only)
    if target is None:
        return name, None, None, None
    game = decoded(loader, log, seat)
    i = target['index']
    obs = np.asarray(game.take_obs()[i], dtype=np.float32)
    mask = np.asarray(game.take_masks()[i], dtype=bool)
    target['seat'] = seat
    return name, target, obs, mask


def agrees(row, target):
    """Whether the replay landed on the decision the row describes."""
    for key in SAME:
        if int(row[key]) != int(target[key]):
            return f'{key} {row[key]} became {target[key]}'
    for key in CLOSE:
        if abs(float(row[key]) - float(target[key])) > 1e-6:
            return f'{key} {row[key]:.9f} became {target[key]:.9f}'
    return None


def features(brain, obs, device, batch=64):
    """The trunk's 1024 numbers for each state, in the precision it was measured in."""
    out = []
    with torch.inference_mode():
        for lo in range(0, len(obs), batch):
            x = torch.as_tensor(np.stack(obs[lo:lo + batch]), device=device)
            out.append(brain(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1024), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True,
                    help='a deviations.json; its own args say how to replay it')
    ap.add_argument('--policy', default=None,
                    help='the policy to replay with. Defaults to the one recorded '
                         'in the results, and must be that one for the logs to match')
    ap.add_argument('--out', default=None, help='defaults to features.npz beside the results')
    ap.add_argument('--blocks', type=int, default=0, help='0 replays every block')
    ap.add_argument('--device', default=None)
    ap.add_argument('--jobs', type=int, default=0)
    args = ap.parse_args()

    # Before CUDA and before torch has threads for a child to deadlock on.
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 2)
    pool = start_workers(jobs)

    with open(args.results) as f:
        saved = json.load(f)
    was = saved['args']
    rows = saved['rows']
    by_block = defaultdict(dict)
    for r in rows:
        by_block[r['block']][r['log']] = r
    blocks = args.blocks or was['blocks']

    policy = args.policy or was['policy']
    device = torch.device(args.device or was.get('device') or config['control']['device'])
    brain, head, version, temperature = load_policy(policy)
    brain, head = brain.to(device), head.to(device)
    logging.info(f'{policy}: v{version}, play temperature {temperature}; replaying the '
                 f'baseline arm of {args.results} across {jobs} processes')

    def engine(name):
        return MortalEngine(brain, head, is_oracle=False, version=version, device=device,
                            enable_amp=was['amp'], enable_rule_based_agari_guard=False,
                            name=name, boltzmann_epsilon=0.)

    champion = engine('champion')
    out = path.join(path.dirname(args.results), 'features.npz') if args.out is None else args.out
    base_dir = path.join(was['out'], 'replay-base')

    kept, dropped = [], []
    for block in range(blocks):
        want = by_block.get(block)
        if not want:
            continue
        first = was['seed_start'] + block * was['seeds']
        names = play(engine(CHALLENGER), champion, (first, was['key']), was['seeds'], base_dir)
        missing = set(want) - set(names)
        if missing:
            raise SystemExit(f'block {block}: the replay produced no {sorted(missing)[:3]}; '
                             f'the seeds or the policy are not the ones that were measured')

        wanted = [(version, base_dir, name,
                   [was['rng'], block, digest64(name) % (1 << 63)], was['pick'],
                   want[name]['rule'], was.get('only', 'all'))
                  for name in sorted(want)]
        obs_here, rows_here = [], []
        for name, target, obs, mask in spread(pool, _feature_job, wanted):
            row = want[name]
            if target is None:
                dropped.append((name, 'the replay found no decision to fork'))
                continue
            if why := agrees(row, target):
                dropped.append((name, why))
                continue
            obs_here.append(obs)
            rows_here.append((row, mask))

        phi = features(brain, obs_here, device)
        for (row, mask), vec in zip(rows_here, phi):
            kept.append((row, mask, vec))
        logging.info(f'block {block}: {len(rows_here)} of {len(want)} states recovered, '
                     f'{len(dropped)} dropped so far')

    if not kept:
        raise SystemExit('nothing was recovered')
    if dropped:
        logging.warning(f'{len(dropped)} states did not come back the same; '
                        f'first few: {dropped[:3]}')

    rules = sorted({r['rule'] for r, _, _ in kept})
    np.savez_compressed(
        out,
        phi=np.stack([v for _, _, v in kept]).astype(np.float32),
        mask=np.stack([m for _, m, _ in kept]),
        label=np.array([r['kyoku_fork'] - r['kyoku_base'] for r, _, _ in kept], np.float32),
        placement=np.array([r['rank_fork'] - r['rank_base'] for r, _, _ in kept], np.float32),
        argmax=np.array([r['argmax'] for r, _, _ in kept], np.int16),
        forced=np.array([r['forced'] for r, _, _ in kept], np.int16),
        forced_rank=np.array([r['forced_rank'] for r, _, _ in kept], np.int16),
        legal=np.array([r['legal'] for r, _, _ in kept], np.int16),
        pass_legal=np.array([r.get('pass_legal', False) for r, _, _ in kept]),
        p_argmax=np.array([r['p_argmax'] for r, _, _ in kept], np.float32),
        p_forced=np.array([r['p_forced'] for r, _, _ in kept], np.float32),
        kyoku=np.array([r['kyoku'] for r, _, _ in kept], np.int16),
        block=np.array([r['block'] for r, _, _ in kept], np.int16),
        rule=np.array([rules.index(r['rule']) for r, _, _ in kept], np.int16),
        rule_names=np.array(rules),
        dropped=np.array(len(dropped)),
    )
    print(f'{len(kept):,} of {len(rows):,} states recovered and encoded, '
          f'{len(dropped)} dropped')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
