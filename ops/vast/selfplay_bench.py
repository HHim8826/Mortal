"""How fast this box plays self-play games, and how that changes as it is split.

Online training consumes games, so this rate is the ceiling on its steps/s --
nothing else about the setup matters if the games do not arrive. One arena
leaves most of the box idle: it advances every game in flight on a single
thread, then encodes the states that must act in parallel, then runs one
forward for all of them, and through the first and last of those the cores have
nothing to do. An evaluation measured 14 of 56 cores busy and the GPU running a
kernel a fifth of the time.

So the question is how many arenas to run at once, and how many encoding
threads to give each. This answers it by playing real games with the real
model, without a server or a trainer in the way.

    cd /root/Mortal/mortal
    MORTAL_CFG=config.online.toml /root/venv/bin/python /root/selfplay_bench.py \\
        --arenas 2 --rayon 14 --games 400 --device cuda:1

Reports hanchans/s, and what that is worth in training steps: a hanchan gives
the trainee about 128 instances, so a 1024 batch costs eight of them.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, '/root/Mortal/mortal')
os.environ.setdefault('MORTAL_CFG', 'config.online.toml')

INSTANCES_PER_HANCHAN = 128     # measured: 40 hanchans gave 5,120 instances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arenas', type=int, default=1)
    ap.add_argument('--rayon', type=int, default=0, help='0 leaves it alone')
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--weights', default='logs/v4/best.pth')
    # Who the three opponents are, which is a throughput question as much as a
    # training one: v4's observation carries the expected-value solve that v3's
    # does not, so three v4 champions cost far more to encode than three v3
    # ones, and the trainee's own seat is the same either way.
    ap.add_argument('--champion', default=None, help='defaults to --weights')
    args = ap.parse_args()

    if args.rayon:
        # Rayon fixes its pool size the first time it is used, so this has to
        # happen before libriichi is imported, let alone asked to play.
        os.environ['RAYON_NUM_THREADS'] = str(args.rayon)

    from config import config
    from model import Brain, DQN

    # By default both sides are the same model: the number being measured is how
    # fast games come out, not who wins them.
    config['baseline']['train']['state_file'] = args.champion or args.weights
    config['baseline']['train']['device'] = args.device
    profile = config['train_play']['default']
    profile['games'] = args.games
    profile['log_dir'] = 'challenger/bench'
    profile['arenas'] = args.arenas

    from player import TrainPlayer

    device = torch.device(args.device)
    state = torch.load(args.weights, weights_only=True, map_location='cpu')
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    mortal = Brain(version=version,
                   conv_channels=cfg['resnet']['conv_channels'],
                   num_blocks=cfg['resnet']['num_blocks']).eval().to(device)
    dqn = DQN(version=version).eval().to(device)
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    player = TrainPlayer()
    start = time.time()
    rankings, files = player.train_play(mortal, dqn, device)
    wall = time.time() - start

    games = int(sum(rankings))
    rate = games / wall
    print(f'arenas {args.arenas}, rayon {os.environ.get("RAYON_NUM_THREADS", "default")}: '
          f'{games:,} hanchans in {wall:.0f} s = {rate:.2f} hanchans/s '
          f'({rate * INSTANCES_PER_HANCHAN:,.0f} instances/s, '
          f'{rate * INSTANCES_PER_HANCHAN / 1024:.2f} steps/s at batch 1024); '
          f'{len(files):,} logs written; champion '
          f'{args.champion or args.weights}', flush=True)


if __name__ == '__main__':
    main()
