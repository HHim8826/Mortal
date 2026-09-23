"""Train the hand-reading model, and check it against the rules it must beat.

    cd mortal
    MORTAL_CFG=config.vast.toml python -m tenpai.train_tenpai \\
        --globs '/root/hf-dataset/data/*.parquet' --steps 20000

Accuracy is not the test here. What goes on a screen is a percentage, so the
test is whether the percentage is true: among the tiles it calls 20%, about a
fifth should turn out to be waits. That is what the Brier score and the
reliability table below measure, and what the plain cross-entropy in
tenpai_net trains for.

Nor is the overall number the test. Most positions are obvious -- nobody is
close to tenpai, or the only tenpai hand has declared riichi and the tile is
one of its own discards -- and a model that only knew genbutsu and suji would
already score well on the average of those. So validation splits out the
places where the old rules have nothing to say:

    no riichi         who is tenpai at all, which suji cannot begin to answer
    called hands      no discards to read suji off
    riichi, non-suji  the tiles a rule can only shrug at, and rank among them

Beating the rule baseline on that third line is the whole point. If it does
not, this model should not ship, and the number it prints would be worse than
the honest "no idea" the rules give.
"""
import argparse
import logging
import os
import time
from glob import glob
from os import path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import prelude                                          # noqa: F401
from tenpai.tenpai_data import (CONTEXT_WIDTH, DISCARDED, FUURO, RIICHI, SUJI,
                         TILE_KINDS, TURN, TenpaiDataset)
from tenpai.tenpai_net import TenpaiNet, losses

# Empirical rates are taken in these buckets: has this seat declared riichi,
# is the tile one of its own discards, is it suji of them, and how late is it.
TURN_BANDS = (6 / 18, 12 / 18)


def bucket(context):
    """A bucket index per (sample, seat, tile), as the old reads would sort them."""
    riichi = context[:, :, RIICHI, None]
    turn = context[:, :, TURN, None]
    genbutsu = context[:, :, DISCARDED:DISCARDED + TILE_KINDS]
    suji = context[:, :, SUJI:SUJI + TILE_KINDS]
    band = (turn > TURN_BANDS[0]).astype(np.int64) + (turn > TURN_BANDS[1]).astype(np.int64)
    return (((riichi.astype(np.int64) * 2 + genbutsu.astype(np.int64)) * 2
             + suji.astype(np.int64)) * 3 + band)


N_BUCKETS = 2 * 2 * 2 * 3


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p, y, bins=10):
    """Predicted against observed, in bins. The table that says if 20% is 20%."""
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if in_bin.sum() >= 50:
            rows.append((lo, hi, float(p[in_bin].mean()), float(y[in_bin].mean()),
                         int(in_bin.sum())))
    return rows


def collate(samples):
    obs, tenpai, waits, any_wait, furiten, context = zip(*samples)
    as_t = lambda xs: torch.from_numpy(np.stack(xs))     # noqa: E731
    return {'obs': as_t(obs), 'tenpai': as_t(tenpai), 'waits': as_t(waits),
            'any_wait': as_t(any_wait), 'furiten': as_t(furiten),
            'context': as_t(context)}


def row_groups(patterns):
    import pyarrow.parquet as pq
    out = []
    for pattern in patterns:
        for shard in sorted(glob(pattern)):
            for group in range(pq.ParquetFile(shard).num_row_groups):
                out.append((shard, group))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--globs', nargs='+', default=['/root/hf-dataset/data/*.parquet'])
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--keep-prob', type=float, default=0.25)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--val-groups', type=int, default=8)
    ap.add_argument('--val-batches', type=int, default=60)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='logs/tenpai/tenpai.pth')
    ap.add_argument('--validate-only', default=None,
                    help='score a saved model instead of training one')
    args = ap.parse_args()

    # Before anything long runs, not after. The first save is the one at the
    # end, so a missing parent directory -- which the default `logs/tenpai`
    # is, in a fresh checkout -- surfaced as a torch.save failure with the
    # whole training behind it and nothing written down.
    out_dir = path.dirname(path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    probe = path.join(out_dir, '.writable')
    with open(probe, 'w'):
        pass
    os.remove(probe)

    device = torch.device(args.device)
    groups = row_groups(args.globs)
    if len(groups) <= args.val_groups:
        raise SystemExit(f'only {len(groups)} row groups found')
    # Held out whole: a row group holds thousands of games, and no game -- so
    # no position, and no other seat's view of it -- can be on both sides.
    rng = np.random.default_rng(20260912)
    order = rng.permutation(len(groups))
    val = [groups[i] for i in order[:args.val_groups]]
    train = [groups[i] for i in order[args.val_groups:]]
    logging.info(f'{len(train):,} row groups to train on, {len(val)} held out')

    def loader(part, seed, workers):
        data = TenpaiDataset(part, keep_prob=args.keep_prob, seed=seed)
        return DataLoader(data, batch_size=args.batch_size, num_workers=workers,
                          collate_fn=collate, drop_last=True,
                          **({'prefetch_factor': 4} if workers else {}))

    net = TenpaiNet(TenpaiDataset(train).channels, args.channels, args.blocks).to(device)
    params = sum(p.numel() for p in net.parameters())
    logging.info(f'{params:,} parameters')

    if args.validate_only:
        saved = torch.load(args.validate_only, weights_only=True, map_location=device)
        net.load_state_dict(saved['model'])
        # The same held-out groups and the same rule rates the run was scored
        # with, since the split is seeded and the rates travel in the file.
        validate(net, loader(val, seed=1, workers=min(args.workers, 4)),
                 np.asarray(saved['baseline']), device, args.val_batches)
        return
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, args.lr / 20)
    scaler = torch.amp.GradScaler(device.type)

    # The rule baseline's rates, counted on the training stream as it goes by,
    # so the yardstick never sees the held-out games either.
    hits = np.zeros(N_BUCKETS)
    seen = np.zeros(N_BUCKETS)

    net.train()
    step = 0
    started = time.time()
    while step < args.steps:
        # Each pass is seeded by the step it starts at, so a pass that trains
        # on nothing starts the next one at the same step with the same seed,
        # samples the same positions, and trains on nothing again, for ever:
        # no update, and no save (issue #6). drop_last makes that easy to hit
        # on a small corpus, and each worker drops its own tail, so there being
        # more samples than a batch in total is not enough.
        pass_started_at = step
        for batch in loader(train, seed=step, workers=args.workers):
            gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                   if k != 'context'}
            with torch.autocast(device.type):
                out = net(gpu['obs'])
                total, parts = losses(out, gpu)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()

            b = bucket(batch['context'].numpy()).reshape(-1)
            w = batch['waits'].numpy().reshape(-1)
            np.add.at(seen, b, 1.)
            np.add.at(hits, b, w)

            step += 1
            if step % 200 == 0:
                rate = step / (time.time() - started)
                logging.info(f'step {step:,}/{args.steps:,} ({rate:.1f}/s) '
                             + ' '.join(f'{k} {v.item():.4f}' for k, v in parts.items()))
            if step >= args.steps:
                break
        if step == pass_started_at:
            raise SystemExit(
                f'a whole pass over {len(train):,} training row groups gave no full batch of '
                f'{args.batch_size:,} (step {step:,} of {args.steps:,}). Each of the '
                f'{max(args.workers, 1)} loader worker(s) reads its own share of the groups, '
                f'keeps a position with probability {args.keep_prob}, and drops its own last '
                f'partial batch. Lower --batch-size or --workers, raise --keep-prob, or add data.')

    torch.save({'model': net.state_dict(), 'args': vars(args),
                'baseline': (hits / np.maximum(seen, 1)).tolist()}, args.out)
    logging.info(f'saved {args.out}')
    validate(net, loader(val, seed=1, workers=min(args.workers, 4)),
             hits / np.maximum(seen, 1), device, args.val_batches)


def validate(net, loader, baseline_rates, device, batches):
    net.eval()
    keep = {'p': [], 'y': [], 'base': [], 'riichi': [], 'suji': [], 'genbutsu': [],
            'fuuro': [], 'tenpai_p': [], 'tenpai_y': []}
    with torch.inference_mode():
        for n, batch in enumerate(loader):
            if n >= batches:
                break
            obs = batch['obs'].to(device, non_blocking=True)
            with torch.autocast(device.type):
                out = net(obs)
            context = batch['context'].numpy()
            keep['p'].append(torch.sigmoid(out['waits'].float()).cpu().numpy().reshape(-1))
            keep['y'].append(batch['waits'].numpy().reshape(-1))
            keep['base'].append(baseline_rates[bucket(context)].reshape(-1))
            keep['riichi'].append(np.repeat(context[:, :, RIICHI], TILE_KINDS))
            keep['fuuro'].append(np.repeat(context[:, :, FUURO], TILE_KINDS))
            keep['suji'].append(context[:, :, SUJI:SUJI + TILE_KINDS].reshape(-1))
            keep['genbutsu'].append(
                context[:, :, DISCARDED:DISCARDED + TILE_KINDS].reshape(-1))
            keep['tenpai_p'].append(torch.sigmoid(out['tenpai'].float()).cpu().numpy().reshape(-1))
            keep['tenpai_y'].append(batch['tenpai'].numpy().reshape(-1))
    if not keep['p']:
        # The same drop_last as training: held-out groups too small to fill one
        # batch per worker give nothing, and np.concatenate would say only that
        # it needs at least one array.
        raise SystemExit('the held-out row groups gave no full batch to validate on; '
                         'lower --batch-size or --workers, or raise --val-groups or --keep-prob')
    data = {k: np.concatenate(v) for k, v in keep.items()}

    logging.info(f'validated on {len(data["y"]):,} (seat, tile) pairs')
    subsets = {
        'every tile': np.ones_like(data['y'], dtype=bool),
        'no riichi': data['riichi'] == 0,
        'called hand': data['fuuro'] == 1,
        'riichi, non-suji': (data['riichi'] == 1) & (data['suji'] == 0)
                            & (data['genbutsu'] == 0),
    }
    logging.info(f'{"subset":<18}{"n":>12}{"waits":>8}{"model":>10}{"rules":>10}{"gain":>8}')
    for name, mask in subsets.items():
        if mask.sum() < 1000:
            continue
        p, b, y = data['p'][mask], data['base'][mask], data['y'][mask]
        model_brier, rule_brier = brier(p, y), brier(b, y)
        gain = 100 * (1 - model_brier / rule_brier) if rule_brier else float('nan')
        logging.info(f'{name:<18}{mask.sum():>12,}{y.mean():>8.3f}'
                     f'{model_brier:>10.5f}{rule_brier:>10.5f}{gain:>7.1f}%')

    logging.info(f'tenpai head: brier {brier(data["tenpai_p"], data["tenpai_y"]):.4f}, '
                 f'nll {nll(data["tenpai_p"], data["tenpai_y"]):.4f}, '
                 f'base rate {data["tenpai_y"].mean():.3f}')
    logging.info('reliability of the per-tile waits, predicted vs observed:')
    for lo, hi, mean_p, mean_y, n in reliability(data['p'], data['y']):
        logging.info(f'  {lo:.1f}-{hi:.1f}  said {mean_p:.3f}  was {mean_y:.3f}  n={n:,}')

    # What a Brier score of 0.066 against 0.069 actually buys, in the only
    # terms a player can act on: sort the tiles the rules cannot separate by
    # what the model thinks of them, and see how often each end is a wait.
    # What a Brier of 0.066 against 0.069 actually buys. Brier is dominated by
    # the tiles nobody could call either way; what a player acts on is the
    # ordering, so sort each subset by what each side thinks and look at the
    # ends. Within "riichi, non-suji" the rules have spent everything they know
    # -- all that is left to them is the turn -- so this is the line where the
    # model either reads something or does not.
    logging.info(f'{"ranked within subset":<20}{"safest tenth":>26}{"riskiest tenth":>22}')
    logging.info(f'{"":<20}{"model":>13}{"rules":>13}{"model":>11}{"rules":>11}')
    for name, mask in subsets.items():
        if mask.sum() < 10000:
            continue
        y = data['y'][mask]
        n = len(y)
        cut = max(1, n // 10)
        ends = []
        for scores in (data['p'][mask], data['base'][mask]):
            order = np.argsort(scores, kind='stable')
            ends.append((y[order[:cut]].mean(), y[order[-cut:]].mean()))
        logging.info(f'  {name:<18}{ends[0][0]:>13.4f}{ends[1][0]:>13.4f}'
                     f'{ends[0][1]:>11.4f}{ends[1][1]:>11.4f}')


if __name__ == '__main__':
    main()
