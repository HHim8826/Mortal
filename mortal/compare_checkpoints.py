"""Two checkpoints over one set of fresh walls, paired, with an honest error.

Every strength number in this project so far is an absolute score against
the v3 ruler on 4,000 games -- 1,000 walls dealt four ways -- and at that
size one avg_pt carries a standard error of 1.33 pt and one avg_rank
0.0177. `best_perf` then takes the maximum over every evaluation a run
made, sixteen of them offline and seven online, so the checkpoint the
project uses was chosen as the argmax of that noise and its recorded score
is inflated by roughly the expected maximum of that many draws.

Two things follow, and this does both at once.

The walls are fresh. Selection against one fixed set is what inflates the
number, so a checkpoint's score there is not a measurement of it any more;
on walls nobody selected against it is. The gap between the two is the
winner's curse, in pt.

And the two models play the same fresh walls, so the deal cancels. That is
worth less here than it sounds -- the run's own logs put the paired error
at 0.0154 rank against 0.025 unpaired, a factor of 1.6, because checkpoints
40,000 steps apart disagree on several percent of decisions and pairing
only cancels the deal, not the play. It is still the right comparison: it
asks whether B beats A rather than whether two separately noisy absolute
numbers happen to be ordered.

Both arms run as tracks of one `play_all`, in threads, sharing one champion
on the GPU. One arena leaves most of a box idle -- it advances every game in
flight on one thread, then encodes in parallel, then runs one forward -- so
two fall into each other's gaps and the pair costs far less than twice one.

    python compare_checkpoints.py --seeds 15000 --key 0x5eed \\
        logs/hf-0911/best_ema.pth logs/best_ema.pth
"""
import argparse
import json
import logging
import os
import time
from glob import glob
from os import path

import torch

import prelude                                          # noqa: F401
from config import config
from evaluate import sha256_of
from model import Brain, DQN
from player import TestPlayer

# The placement points every strength number in this project is quoted in.
# `evaluate.py` owns them; the kyoku GRP delta is a different scale and the
# two must never be mixed.
PTS = [90, 45, 0, -135]
# Beside a track's games, what played them. Not *.json: `Stat.from_dir` reads
# every *.json under the directory as a game log.
IDENTITY = 'identity.jsonl'


def load(file):
    """A checkpoint's EMA weights, which is what every test play measured.

    A `best_ema` file holds the average in both slots -- `state['mortal']`
    and `state['ema']['mortal']` are byte for byte the same, so the file
    plays as the EMA on its own -- while an ordinary checkpoint keeps the
    trained weights in the first and the average in the second. Reading
    `ema` when it is there covers both without having to know which kind of
    file this is.
    """
    state = torch.load(file, weights_only=True, map_location='cpu')
    cfg = state['config']
    version = cfg['control']['version']
    brain = Brain(version=version, **cfg['resnet']).eval()
    dqn = DQN(version=version).eval()
    ema = state.get('ema')
    brain.load_state_dict(ema['mortal'] if ema else state['mortal'])
    dqn.load_state_dict(ema['current_dqn'] if ema else state['current_dqn'])
    steps = state.get('steps')
    if steps is None:
        got = state.get('optimizer', {}).get('state', {}).get(0, {}).get('step')
        steps = int(got) if got is not None else None
    return brain, dqn, version, steps, state.get('best_perf'), 'ema' if ema else 'trained'


def identity(file, weights, opponent):
    """What a track's games were played by, as far as the games can depend on it.

    The contents, not the path: a trainer overwrites `best_ema.pth` in place,
    and a path that still names the checkpoint an interrupted comparison was
    playing says nothing about whether it still holds it.
    """
    base = config['baseline']['test']
    return {
        'checkpoint': sha256_of(file),
        'weights': weights,
        'opponent': sha256_of(opponent),
        'opponent_head': base.get('head', 'dqn'),
        'version': config['control']['version'],
    }


def played_by(track_dir):
    try:
        with open(path.join(track_dir, IDENTITY), encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoints', nargs=2, help='A and B; the result is B minus A')
    ap.add_argument('--seeds', type=int, default=15000,
                    help='walls; each is dealt four times, so games are four times this')
    ap.add_argument('--key', default='0x5eed',
                    help='the wall key. Anything but the 0x2000 every evaluation so far '
                         'has used, or this measures the set that was selected against')
    ap.add_argument('--base', type=int, default=10000)
    ap.add_argument('--chunk', type=int, default=1000,
                    help='walls per arena call. Peak memory scales with the games an '
                         'arena carries at once -- 250 walls measured 2.4 GB and 1,000 '
                         'measured 4.4 GB -- so the whole run is played in pieces and '
                         'the logs accumulate. Throughput needs games in flight too '
                         '(4.6 games/s at 250 walls against 5.4 at 1,000), so this is '
                         'the largest piece that fits rather than the smallest')
    ap.add_argument('--resume', action='store_true',
                    help='keep the games already in the track directories and play only '
                         'the range asked for now. `collect` and `paired` both read the '
                         'directory, so a run cut short is finished by pointing --base '
                         'at the first wall it never reached; without this the first '
                         'thing a run does is delete what the last one played')
    ap.add_argument('--opponent', default=None,
                    help='score against this checkpoint instead of the configured ruler. '
                         'A run trains against a frozen copy of itself and is measured '
                         'against something else; if a checkpoint is best-responding to '
                         'the copy rather than improving, it wins here and loses there, '
                         'and only running both says which happened')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    key = int(args.key, 0)
    if key == 0x2000:
        raise SystemExit('0x2000 is the wall set every evaluation in this project has '
                         'used and been selected on; pick another or the winner\'s '
                         'curse is baked into the answer')

    device = torch.device(args.device or config['control']['device'])
    player = TestPlayer(device=device, opponent=args.opponent)
    player.seed_base, player.seed_key = args.base, key

    opponent = args.opponent or config['baseline']['test']['state_file']
    jobs, meta = [], []
    for name, file in zip(('a', 'b'), args.checkpoints):
        brain, dqn, version, steps, best, weights = load(file)
        if version != player.chal_version:
            raise SystemExit(f'{file} is v{version} and the config is '
                             f'v{player.chal_version}; they read different observations')
        jobs.append((brain.to(device).requires_grad_(False),
                     dqn.to(device).requires_grad_(False), name))
        label = path.join(path.basename(path.dirname(file)), path.basename(file))
        meta.append((name, label, steps, best))
        track = player.track_dir(name)
        want = dict(identity(file, weights, opponent), key=key)
        if args.resume:
            have = len(glob(path.join(track, '**', '*.json.gz'), recursive=True))
            had = played_by(track)
            # `paired` matches games by seed and nothing else, so games from
            # another checkpoint pair up with these just as well and the mix
            # comes out labelled as this one. Checked here or not at all.
            if have and had != want:
                raise SystemExit(
                    f'{name}: the {have:,} games in {track} were played by '
                    f'{had or "something this version did not record"}, and this run is '
                    f'{want}. --resume only continues the same checkpoint, weights, '
                    'opponent and key; drop it to start the track over')
            logging.info(f'{name}: keeping {have:,} games already played')
        else:
            player.clear(name)
        os.makedirs(track, exist_ok=True)
        with open(path.join(track, IDENTITY), 'w', encoding='utf-8') as f:
            json.dump(want, f)

    logging.info(f'opponent: {opponent}')
    for name, base, steps, best in meta:
        logging.info(f'{name}: {base}, {steps:,} steps, recorded best {best}'
                     if steps else f'{name}: {base}, recorded best {best}')
    logging.info(f'{args.seeds:,} walls at key {key:#x} = {args.seeds * 4:,} games, '
                 f'both arms on the same walls')

    started = time.monotonic()
    done = 0
    while done < args.seeds:
        here = min(args.chunk, args.seeds - done)
        player.seed_base = args.base + done
        player.play_all(here, jobs, device)
        done += here
        took = time.monotonic() - started
        free = int(os.popen("free -m | awk '/Mem:/{print $7}'").read() or 0)
        note = ''
        if done < args.seeds:
            diff, se, _, walls = player.paired('b', against='a')
            note = f', so far {diff:+.4f} +- {se:.4f} over {walls:,} walls'
            eta = took / done * (args.seeds - done) / 60
            note += f', {eta:.0f} min left'
        logging.info(f'{done:,}/{args.seeds:,} walls, {done * 8 / took:.1f} games/s, '
                     f'{free:,} MB free{note}')
    took = time.monotonic() - started
    logging.info(f'played in {took / 60:.1f} min, '
                 f'{args.seeds * 8 / took:.1f} games/s over both arms')

    print()
    stats = {}
    for name, base, steps, best in meta:
        stat = player.collect(name)
        avg_pt = stat.avg_pt(PTS)
        stats[name] = (base, stat, avg_pt, best)
        games = stat.game
        print(f'{base:<26} {games:>7,} games   rank {stat.avg_rank:.4f}   '
              f'pt {avg_pt:+.3f}   recorded best {best["avg_pt"]:+.3f} / '
              f'{best["avg_rank"]:.4f}')

    # The error is over walls: a seed is played four times off one deal and
    # those four are not four independent draws.
    diff, se, games, walls = player.paired('b', against='a')
    (base_a, _, _, _), (base_b, _, _, _) = stats['a'], stats['b']
    print()
    print(f'paired over {walls:,} walls ({games:,} games), negative is better:')
    print(f'  {base_b} minus {base_a}: rank {diff:+.4f} +- {se:.4f}  '
          f'({abs(diff / se) if se else 0:.1f} se)')

    print()
    print('recorded best against this run, in pt. The recorded figure is the maximum')
    print('over a run\'s evaluations on one fixed wall set, so it is inflated by some')
    print('amount; this one is a single evaluation on fresh walls, so it carries its')
    print('own sampling error, about 1.33 pt at 4,000 games and less here. The')
    print('difference is both together and is not an estimate of the selection bias:')
    for name in ('a', 'b'):
        base, stat, avg_pt, best = stats[name]
        print(f'  {base:<24} recorded {best["avg_pt"]:+.3f}, here {avg_pt:+.3f}, '
              f'difference {best["avg_pt"] - avg_pt:+.3f} pt')


if __name__ == '__main__':
    main()
