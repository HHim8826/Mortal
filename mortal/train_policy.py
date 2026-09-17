"""Give a trained v4 model a policy of its own and a critic of the whole game.

    cd mortal
    python train_policy.py --from logs/hf-0911/best_ema.pth \\
        --globs '/root/hf-dataset/data/*.parquet' --steps 20000

Mortal decides by argmax of Q, and that Q is doing three jobs: estimating the
return, carrying the offline CQL term's shaping towards human actions, and
serving -- divided by a temperature nobody calibrated -- as the scale
exploration samples on. A policy gradient needs a distribution it can move
without moving a value estimate, so this trains two new heads on the trunk
that is already there:

    policy   the 46 action logits, distilled from the teacher's own masked
             softmax(Q / T), so the new head starts out playing the same game
             as the model it came from rather than from nothing
    critic   the probability of finishing 1st, 2nd, 3rd or 4th in the hanchan,
             from the placements the corpus actually reached

The trunk is frozen here, which makes this cheap -- one forward, no gradient
through 40 residual blocks -- and makes the comparison clean: teacher and
student read the same features, so a difference between them is in the heads.
It also means the distillation can be exact. A v4 Q is v + a - mean(a) over
the legal actions, and the softmax of that is the softmax of a alone, so a
linear policy head can match the teacher tile for tile. Whether it does is
what `argmax agrees` reports.

The unfreezing, and the policy gradient this is the starting point for, come
later: what has to be true first is that the policy plays as well as the Q it
replaces, and that the critic's four numbers are worth using as a baseline.
"""
import argparse
import logging
import os
import time
from glob import glob
from os import path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import prelude                                          # noqa: F401
from config import config
from dataloader import FileDatasetsIter, worker_init_fn
from model import Brain, DQN, PolicyHead, RankCritic


def row_groups(patterns):
    """Every (shard, row group) the globs name, the unit the loader reads."""
    import pyarrow.parquet as pq
    out = []
    for pattern in patterns:
        for shard in sorted(glob(pattern, recursive=True)):
            for group in range(pq.ParquetFile(shard).num_row_groups):
                out.append((shard, group))
    return out


def load_teacher(file, device):
    state = torch.load(file, weights_only=True, map_location='cpu')
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    if version != 4:
        raise SystemExit(f'{file} is a v{version} model; this wants v4')
    brain = Brain(version=version, conv_channels=cfg['resnet']['conv_channels'],
                  num_blocks=cfg['resnet']['num_blocks']).eval()
    dqn = DQN(version=version).eval()
    brain.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    brain.requires_grad_(False)
    dqn.requires_grad_(False)
    return brain.to(device), dqn.to(device), state, cfg


def distill_loss(student_logits, teacher_logits, temperature):
    """Cross entropy against the teacher's masked softmax, and the KL in it.

    Illegal actions are -inf on both sides, so they take no probability and
    contribute nothing; the sum is over the legal ones alone.

    Both are reported because only one of them can reach zero. The cross
    entropy cannot go below the teacher's own entropy, which with eight or
    nine legal actions is most of what it measures and barely moves while the
    student learns. The KL is what is left after that floor: the distance
    still to cover.
    """
    tempered = teacher_logits / temperature
    target = tempered.softmax(-1)
    # An illegal action is -inf on both sides: probability 0 and log -inf,
    # whose product is a nan rather than the 0 it stands for.
    def against(logits):
        logp = logits.log_softmax(-1)
        return -torch.where(target > 0, target * logp, torch.zeros_like(logp)).sum(-1)
    cross = against(student_logits)
    return cross.mean(), (cross - against(tempered)).mean()


def entropy_of(logits):
    logp = logits.log_softmax(-1)
    p = logp.exp()
    return -torch.where(p > 0, p * logp, torch.zeros_like(logp)).sum(-1)


def explained_variance(pred, actual):
    """1 - Var(actual - pred) / Var(actual); 0 is no better than the mean."""
    var = actual.var()
    return float('nan') if var == 0 else float(1 - (actual - pred).var() / var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='teacher', required=True,
                    help='the v4 checkpoint whose trunk and Q are the starting point')
    ap.add_argument('--globs', nargs='+', default=None,
                    help='parquet shards; defaults to the config dataset')
    ap.add_argument('--out', default='logs/policy/policy.pth')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--temperature', type=float, default=1.0,
                    help='the teacher softmax the policy is distilled from')
    ap.add_argument('--critic-weight', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--file-batch-size', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--log-every', type=int, default=200)
    ap.add_argument('--save-every', type=int, default=2000)
    args = ap.parse_args()

    out_dir = path.dirname(path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device or config['control']['device'])
    brain, dqn, teacher_state, cfg = load_teacher(args.teacher, device)
    logging.info(f'teacher {args.teacher}: step {teacher_state["steps"]:,}, '
                 f'{cfg["resnet"]["conv_channels"]}x{cfg["resnet"]["num_blocks"]}, '
                 f'best {teacher_state.get("best_perf")}')

    pts = tuple(config['env']['pts'])
    policy = PolicyHead(version=4).to(device)
    critic = RankCritic(pts=pts).to(device)
    logging.info(f'policy {sum(p.numel() for p in policy.parameters()):,} parameters, '
                 f'critic {sum(p.numel() for p in critic.parameters()):,}, '
                 f'placement utility {pts}')

    file_list = row_groups(args.globs) if args.globs else row_groups(
        config['dataset'].get('parquet_globs') or [])
    if not file_list:
        raise SystemExit('no parquet row groups to read; pass --globs')
    logging.info(f'{len(file_list):,} row groups')

    workers = args.workers if args.workers is not None else config['dataset']['num_workers']
    file_batch_size = (args.file_batch_size if args.file_batch_size is not None
                       else config['dataset'].get('file_batch_size', 20))
    data = FileDatasetsIter(
        version = 4,
        file_list = file_list,
        pts = list(pts),
        parquet = True,
        file_batch_size = file_batch_size,
        num_epochs = 10 ** 6,       # stopped by --steps, not by the corpus
        final_rank = True,
        # Neither head learns from the per-kyoku return, and it costs a GRP
        # forward for every game decoded.
        skip_rewards = True,
    )
    loader_kwargs = {}
    if workers > 0:
        loader_kwargs['prefetch_factor'] = config['dataset'].get('prefetch_factor', 2)
        loader_kwargs['in_order'] = config['dataset'].get('in_order', True)
    batches = DataLoader(
        dataset = data,
        batch_size = args.batch_size,
        drop_last = True,
        num_workers = workers,
        pin_memory = True,
        worker_init_fn = worker_init_fn,
        **loader_kwargs,
    )
    loader = iter(batches)

    optimizer = optim.AdamW(list(policy.parameters()) + list(critic.parameters()),
                            lr=args.lr, weight_decay=0.01)
    schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, args.lr / 20)
    scaler = torch.amp.GradScaler(device.type)
    cross_entropy = nn.CrossEntropyLoss()

    def save(step):
        torch.save({
            'policy': policy.state_dict(),
            'critic': critic.state_dict(),
            # The trunk and the Q it was distilled from, unchanged, so the file
            # plays on its own and can be compared with its teacher directly.
            'mortal': brain.state_dict(),
            'current_dqn': dqn.state_dict(),
            'config': cfg,
            'steps': step,
            'pts': list(pts),
            'distilled_from': {'file': path.abspath(args.teacher),
                               'steps': teacher_state['steps'],
                               'temperature': args.temperature},
            'args': vars(args),
        }, args.out)

    stats = {}
    started = time.time()
    for step in range(1, args.steps + 1):
        obs, actions, masks, _steps_to_done, _kyoku_rewards, _player_ranks, final_rank = next(loader)
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
        masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
        final_rank = final_rank.to(dtype=torch.int64, device=device, non_blocking=True)

        with torch.autocast(device.type):
            # The trunk is frozen, so the teacher's features are the student's:
            # one forward serves both, and no gradient flows into it.
            with torch.no_grad():
                phi = brain(obs)
                teacher_logits = dqn(phi, masks)
            student_logits = policy(phi, masks)
            rank_logits = critic(phi)
            policy_loss, kl = distill_loss(student_logits, teacher_logits, args.temperature)
            critic_loss = cross_entropy(rank_logits, final_rank)
            loss = policy_loss + args.critic_weight * critic_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        schedule.step()

        with torch.inference_mode():
            agree = (student_logits.argmax(-1) == teacher_logits.argmax(-1)).float().mean()
            human = (student_logits.argmax(-1) == actions).float().mean()
            value = critic.value(rank_logits.float())
            actual = critic.pts[final_rank]
            for key, val in (('policy_loss', policy_loss), ('kl', kl), ('critic_loss', critic_loss),
                             ('argmax_agrees', agree), ('argmax_is_human', human),
                             ('entropy', entropy_of(student_logits.float()).mean()),
                             ('legal_actions', masks.sum(-1).float().mean())):
                stats[key] = stats.get(key, 0.) + float(val)
            stats['ev'] = stats.get('ev', 0.) + explained_variance(value, actual)

        if step % args.log_every == 0:
            rate = step / (time.time() - started)
            means = {k: v / args.log_every for k, v in stats.items()}
            stats.clear()
            logging.info(
                f'step {step:,}/{args.steps:,} ({rate:.1f}/s) '
                f'distill {means["policy_loss"]:.4f} (kl {means["kl"]:.4f}) '
                f'critic {means["critic_loss"]:.4f} | '
                f'argmax agrees {means["argmax_agrees"]:.3f}, is human {means["argmax_is_human"]:.3f}, '
                f'entropy {means["entropy"]:.3f} over {means["legal_actions"]:.1f} legal, '
                f'critic ev {means["ev"]:+.3f}')
        if step % args.save_every == 0 or step == args.steps:
            save(step)
            logging.info(f'saved {args.out} at step {step:,}')

    # Without workers the decoding happens in this process, in a thread that
    # sits on a semaphore between batches. Closing the generator lets it finish
    # instead of being killed on the way out, which aborts the interpreter.
    del loader
    if data.iterator is not None:
        data.iterator.close()


if __name__ == '__main__':
    main()
