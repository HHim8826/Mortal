"""Move the policy with the human corpus, weighted by how the kyoku went.

    cd mortal
    python train_awr.py --from logs/policy/policy-t0.05.pth \
        --globs '/root/hf-dataset/data/*.parquet' --steps 20000

Phase 3 established that online PPO is not broken here, it is slow: the
advantage favours the argmax by 9.1 se, the gradient's consistent direction is
exactly that, and raising the learning rate thirty-fold made each update four
times more effective -- and still left 17,000 updates, some forty hours of
self-play, between here and closing the gap that sampling opens. Every update
has to squeeze a direction out of sixteen thousand decisions whose returns are
mostly luck.

That gap is also smaller than the phase assumed. This docstring used to put it
at 0.023 of rank; that number was contaminated from the online560-offline520
comparison and nothing ever measured it. What phase 2 measured is that
sampling at T=0.05 costs -1.13 +- 1.53 pt against the same policy's argmax,
which is about 0.011 of rank and is not distinguishable from zero.

Advantage weighted regression does not have that problem. It is behaviour
cloning where each human action is weighted by exp(A / beta): every one of the
corpus's 1.66M hanchans contributes a target with a known weight, none of it
waits on self-play, and nothing depends on the signal-to-noise of a single
batch. It is how Nitasurin/Mortal-Policy starts the same transition on the
same v4 before letting PPO refine it -- the step this project skipped.

The advantage is the kyoku's own return against what the value head expected
of the state, so a weight above one means the hand went better than the
position deserved, and the action that human chose is pulled towards. The
reference weights by the raw return instead, with no baseline; `--baseline
none` does that, and the difference is whether a quiet kyoku in a winning
position counts as evidence.

The trunk stays frozen, as it has throughout: one forward serves the policy
and the value, and what changes is the decision rule rather than the
representation.
"""
import argparse
import logging
import os
import time
from glob import glob
from os import path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import prelude                                          # noqa: F401
from config import config
from dataloader import FileDatasetsIter, worker_init_fn
from model import Brain, DQN, KyokuValue, PolicyHead, RankCritic


def row_groups(patterns):
    """Every (shard, row group) the globs name, the unit the loader reads."""
    import pyarrow.parquet as pq
    out = []
    for pattern in patterns:
        for shard in sorted(glob(pattern, recursive=True)):
            for group in range(pq.ParquetFile(shard).num_row_groups):
                out.append((shard, group))
    return out


def load_start(file, device):
    """The policy this departs from, and the baseline its advantage is against."""
    state = torch.load(file, weights_only=True, map_location='cpu')
    cfg = state['config']
    if cfg['control'].get('version', 1) != 4:
        raise SystemExit(f'{file} is not a v4 model')
    brain = Brain(version=4, conv_channels=cfg['resnet']['conv_channels'],
                  num_blocks=cfg['resnet']['num_blocks']).eval()
    brain.load_state_dict(state['mortal'])
    brain.requires_grad_(False)
    policy = PolicyHead(version=4)
    policy.load_state_dict(state['policy'])
    critic = RankCritic(pts=tuple(state.get('pts', config['env']['pts'])))
    if 'critic' in state:
        critic.load_state_dict(state['critic'])
    if 'value' in state:
        value = KyokuValue()
        value.load_state_dict(state['value'])
    else:
        teacher = DQN(version=4)
        teacher.load_state_dict(state['current_dqn'])
        value = KyokuValue.from_dueling(teacher)
    return (brain.to(device), policy.to(device), critic.to(device),
            value.to(device), state, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start', required=True,
                    help='the policy checkpoint this departs from')
    ap.add_argument('--globs', nargs='+', default=None,
                    help='parquet shards; defaults to the config dataset')
    ap.add_argument('--out', default='logs/awr/policy.pth')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--beta', type=float, default=1.0,
                    help='temperature of the weight; larger is closer to plain cloning')
    ap.add_argument('--weight-clip', type=float, default=100.,
                    help='the largest weight a single sample may carry')
    ap.add_argument('--baseline', choices=('value', 'none'), default='value',
                    help="'none' weights by the raw return, as the reference does")
    ap.add_argument('--v-coef', type=float, default=0.5)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--file-batch-size', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--log-every', type=int, default=250)
    ap.add_argument('--save-every', type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
    device = torch.device(args.device or config['control']['device'])
    brain, policy, critic, value_head, state, cfg = load_start(args.start, device)
    start_policy = PolicyHead(version=4).to(device).eval().requires_grad_(False)
    start_policy.load_state_dict(policy.state_dict())
    logging.info(f'start {args.start}: play temperature {state.get("play_temperature")}, '
                 f'beta {args.beta}, weights clipped at {args.weight_clip:g}, '
                 f'baseline {args.baseline}')

    file_list = row_groups(args.globs) if args.globs else row_groups(
        config['dataset'].get('parquet_globs') or [])
    if not file_list:
        raise SystemExit('no parquet row groups to read; pass --globs')
    logging.info(f'{len(file_list):,} row groups')

    workers = args.workers if args.workers is not None else config['dataset']['num_workers']
    file_batch_size = (args.file_batch_size if args.file_batch_size is not None
                       else config['dataset'].get('file_batch_size', 20))
    player_names = set()
    for filename in config['dataset'].get('player_names_files') or []:
        with open(filename) as f:
            player_names.update(line.strip() for line in f if line.strip())
    data = FileDatasetsIter(
        version=4, file_list=file_list, pts=list(config['env']['pts']),
        parquet=True, file_batch_size=file_batch_size,
        player_names=sorted(player_names) or None,
        num_epochs=10 ** 6, final_rank=True,
    )
    loader_kwargs = {}
    if workers > 0:
        loader_kwargs['prefetch_factor'] = config['dataset'].get('prefetch_factor', 2)
        loader_kwargs['in_order'] = config['dataset'].get('in_order', True)
    batches = DataLoader(dataset=data, batch_size=args.batch_size, drop_last=True,
                         num_workers=workers, pin_memory=True,
                         worker_init_fn=worker_init_fn, **loader_kwargs)
    loader = iter(batches)

    optimizer = optim.AdamW(list(policy.parameters()) + list(value_head.parameters()),
                            lr=args.lr, weight_decay=0.)
    schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, args.lr / 20)
    scaler = torch.amp.GradScaler(device.type)
    huber = nn.SmoothL1Loss()
    gamma = config['env']['gamma']

    def save(step):
        torch.save({
            'policy': policy.state_dict(),
            'critic': critic.state_dict(),
            'value': value_head.state_dict(),
            'mortal': brain.state_dict(),
            'current_dqn': state['current_dqn'],
            'config': cfg,
            'pts': critic.pts.tolist(),
            'play_temperature': state.get('play_temperature'),
            'steps': step,
            'started_from': path.abspath(args.start),
            'args': vars(args),
        }, args.out)

    stats, started = {}, time.time()
    for step in range(1, args.steps + 1):
        obs, actions, masks, steps_to_done, kyoku_rewards, _ranks, _final = next(loader)
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
        masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
        steps_to_done = steps_to_done.to(dtype=torch.int64, device=device, non_blocking=True)
        kyoku_rewards = kyoku_rewards.to(dtype=torch.float32, device=device, non_blocking=True)

        with torch.autocast(device.type):
            phi = brain(obs).detach()
            logits = policy(phi, masks).float()
            value = value_head(phi).float()
            paid = gamma ** steps_to_done * kyoku_rewards
            advantage = paid - (value.detach() if args.baseline == 'value' else 0.)
            # exp of an advantage is a weight, and one sample must not be able
            # to become the batch: the reference clips at 100 and so does this.
            weight = (advantage / args.beta).exp().clamp(max=args.weight_clip)
            logp = logits.log_softmax(-1).gather(-1, actions[:, None]).squeeze(-1)
            policy_loss = -(weight * logp).mean()
            value_loss = huber(value, paid)
            loss = policy_loss + args.v_coef * value_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        schedule.step()

        with torch.inference_mode():
            p = logits.softmax(-1)
            safe = logits.log_softmax(-1).masked_fill(~torch.isfinite(logits), 0.)
            ref = start_policy(phi.float(), masks).log_softmax(-1).masked_fill(
                ~torch.isfinite(logits), 0.)
            for key, val in (
                ('policy_loss', policy_loss), ('value_loss', value_loss),
                ('weight', weight.mean()), ('weight_max', weight.max()),
                ('clipped', (weight >= args.weight_clip).float().mean()),
                ('is_human', (logits.argmax(-1) == actions).float().mean()),
                ('entropy', -(p * safe).sum(-1).mean()),
                ('kl_to_start', (p * (safe - ref)).sum(-1).mean()),
                ('advantage', advantage.mean()),
            ):
                stats[key] = stats.get(key, 0.) + float(val)

        if step % args.log_every == 0:
            m = {k: v / args.log_every for k, v in stats.items()}
            stats.clear()
            logging.info(
                f'step {step:,}/{args.steps:,} ({step / (time.time() - started):.1f}/s) '
                f'policy {m["policy_loss"]:+.4f} value {m["value_loss"]:.4f} | '
                f'weight {m["weight"]:.3f} (max {m["weight_max"]:.1f}, '
                f'clipped {m["clipped"]:.2%}), advantage {m["advantage"]:+.3f} | '
                f'argmax is human {m["is_human"]:.3f}, entropy {m["entropy"]:.3f}, '
                f'kl from start {m["kl_to_start"]:.4f}')
        if step % args.save_every == 0 or step == args.steps:
            save(step)
            logging.info(f'saved {args.out} at step {step:,}')

    del loader
    if data.iterator is not None:
        data.iterator.close()


if __name__ == '__main__':
    main()
