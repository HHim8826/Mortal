"""What a checkpoint scores on games it was never trained on.

v4 held nothing out. `train.py` globs every shard, indexes every row group
and splits none of it off, so there is no in-distribution data the run has
not seen and no way to ask whether it was still learning when it stopped.

The corpus is an incomplete scrape, though, and the release archive is a
more complete one of the same lobby and the same years, converted to the
same bytes. The games in the archive that are missing from the corpus are
therefore the one thing that is both unseen and exactly in distribution.
`make_holdout` writes them, and a month-matched sample of games that ARE in
the corpus beside them -- October 2015 is missing 5.5% of its games and
January none, so a flat sample would differ in month as well as in training
and a month effect would arrive looking like a generalisation gap.

The three numbers are the three the run was trained on:

  human action CE   the CQL term, `logsumexp(Q) - Q(a_human)`, which is the
                    masked cross entropy of what the player actually did.
                    The clearest of the three: it is the model's opinion of
                    houou play, on games it has not read.
  value MSE         `0.5 * mse(Q(a_human), gamma^n * kyoku_reward)`.
  next rank CE      the auxiliary head, on the final placement.

What the gap between seen and unseen means:

  still falling on both, gap small   undertrained; train longer
  unseen flat, seen far below it     overfitting; the corpus is the limit
  both flat and close together       capacity; a bigger trunk is the lever

    python validate_supervised.py --holdout /root/holdout \\
        logs/mortal-120000.pth logs/mortal-800k.pth logs/best_ema.pth
"""
import argparse
import logging
from collections import defaultdict
from glob import glob
from os import path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import prelude                                          # noqa: F401
from config import config
from dataloader import FileDatasetsIter, worker_init_fn
from model import AuxNet, Brain, DQN


def shards(holdout, prefix):
    """Row groups of one side of the split, in the order the loader wants."""
    import pyarrow.parquet as pq
    out = []
    for shard in sorted(glob(path.join(holdout, f'{prefix}-*.parquet'))):
        out.extend((shard, rg) for rg in range(pq.ParquetFile(shard).num_row_groups))
    return out


def load(file, device):
    """A checkpoint as the three modules it was saved as.

    Every run in this project saves the modules separately, and the EMA
    keeps its average under its own key, so one loader covers the plain
    checkpoints and the averaged one.

    The shape comes from the checkpoint's own config rather than the
    ambient one. `logs/` holds a v3 file from 2023 beside the v4 ones, and
    a v3 trunk reads a 934x34 observation where v4 reads 1012x34 -- built
    from the wrong config it would either fail to load or, worse, load and
    be scored on an encoding it was never trained for.
    """
    state = torch.load(file, weights_only=True, map_location='cpu')
    cfg = state.get('config') or {}
    version = cfg.get('control', {}).get('version') or config['control']['version']
    resnet = cfg.get('resnet') or config['resnet']
    mortal = Brain(version=version, **resnet)
    dqn = DQN(version=version)
    aux = AuxNet((4,))
    for module, keys in ((mortal, ('mortal', 'mortal_ema')),
                         (dqn, ('current_dqn', 'dqn', 'current_dqn_ema')),
                         (aux, ('aux_net', 'aux_net_ema'))):
        for k in keys:
            if k in state:
                module.load_state_dict(state[k])
                break
        else:
            raise SystemExit(f'{file} has none of {keys}; keys are {sorted(state)}')
    return version, (mortal.eval().to(device).requires_grad_(False),
                     dqn.eval().to(device).requires_grad_(False),
                     aux.eval().to(device).requires_grad_(False))


def score(loaded, file_list, device, version, batches, file_batch):
    """Every checkpoint's three losses, on one pass over the data.

    Decoding is what this costs -- the forward pass leaves the GPU at a
    third and waits -- so the checkpoints are scored inside the batch loop
    rather than each getting a pass of its own. Four passes became two.

    Totals are kept per game, not per decision, because the standard error
    has to be. The ~600 decisions of one hanchan share its deal, its
    opponents and its outcome, so treating them as 900,000 independent
    draws would divide the error by thirty more than it has any right to.
    This is the same mistake phase 3 made, in a different costume.
    """
    data = FileDatasetsIter(
        version=version, file_list=list(file_list), pts=config['env']['pts'],
        # How many row groups are decoded before any of them is consumed,
        # and so what the peak memory is. A v4 observation is 1012x34 and a
        # hanchan holds a few hundred, so a few hundred games is already
        # gigabytes: the corpus's own 4,000-row groups would decode 1,359
        # games at once and take the machine down. `make_holdout` writes
        # small groups and this stays small to match.
        file_batch_size=file_batch,
        # The wall, not the trajectory. `decision_ids` names (game, seat),
        # and the four seats of one hanchan share its deal, so clustering on
        # that counts one hanchan as four independent groups.
        reserve_ratio=0., parquet=True, player_names=[], wall_ids=True,
        num_epochs=1, enable_augmentation=False, augmented_first=False)
    loader = DataLoader(
        dataset=data, batch_size=config['control']['batch_size'],
        drop_last=False, num_workers=0, worker_init_fn=worker_init_fn)
    gamma = config['env']['gamma']
    mse, ce = nn.MSELoss(reduction='none'), nn.CrossEntropyLoss(reduction='none')

    # name -> game id -> [cql, value, rank, agree, decisions]
    per_game = {name: defaultdict(lambda: np.zeros(5)) for name in loaded}
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if batches and i >= batches:
                break
            obs, actions, masks, steps_to_done, kyoku_rewards, ranks = batch[:6]
            games = batch[6]      # the wall, one per hanchan
            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
            ranks = ranks.to(dtype=torch.int64, device=device)
            rows = torch.arange(len(obs), device=device)
            target = (gamma ** steps_to_done * kyoku_rewards).to(torch.float32)
            keys = [g if isinstance(g, str) else str(g) for g in games]

            for name, (mortal, dqn, aux) in loaded.items():
                phi = mortal(obs)
                q_out = dqn(phi, masks)
                q = q_out[rows, actions]
                each = torch.stack([
                    q_out.logsumexp(-1) - q,
                    0.5 * mse(q, target),
                    ce(aux(phi)[0], ranks),
                    (q_out.argmax(-1) == actions).to(torch.float32),
                ], dim=1).double().cpu().numpy()
                table = per_game[name]
                for k, row in zip(keys, each):
                    slot = table[k]
                    slot[:4] += row
                    slot[4] += 1
    return per_game


def summarise(table):
    """Per-game means, and a standard error that counts games not decisions."""
    got = np.array(list(table.values()))
    weight = got[:, 4]
    means = got[:, :4] / weight[:, None]
    # Weighted by how many decisions each game contributed, so the point
    # estimate is still the per-decision mean; the error is over games.
    w = weight / weight.sum()
    mean = (means * w[:, None]).sum(0)
    var = (w[:, None] * (means - mean) ** 2).sum(0) * (w ** 2).sum() / (1 - (w ** 2).sum())
    return mean, np.sqrt(var), len(got), int(weight.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--holdout', default='/root/holdout')
    ap.add_argument('--batches', type=int, default=0, help='0 reads everything')
    ap.add_argument('--file-batch', type=int, default=4,
                    help='row groups decoded at once; this times the row group '
                         'size is the peak, in games held decoded')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = torch.device(args.device or config['control']['device'])
    sides = {name: shards(args.holdout, name) for name in ('unseen', 'seen')}
    for name, got in sides.items():
        if not got:
            raise SystemExit(f'no {name}-*.parquet in {args.holdout}')
        logging.info(f'{name}: {len(got)} row groups')

    loaded, versions = {}, set()
    for file in args.checkpoints:
        version, modules = load(file, device)
        loaded[path.basename(file)] = modules
        versions.add(version)
    if len(versions) > 1:
        raise SystemExit(f'checkpoints disagree on the observation version: {versions}. '
                         'They cannot share one pass over the data, and comparing them '
                         'across encodings would not mean anything either.')
    version = versions.pop()

    got = {}
    for side, file_list in sides.items():
        got[side] = score(loaded, file_list, device, version, args.batches,
                          args.file_batch)
        logging.info(f'{side}: scored {len(args.checkpoints)} checkpoints in one pass')

    print()
    print(f'{"checkpoint":<22} {"":7} {"human action CE":>18} {"value MSE":>17} '
          f'{"next rank CE":>17} {"argmax = human":>16}')
    for name in loaded:
        summary = {}
        for side in ('seen', 'unseen'):
            mean, se, games, decisions = summarise(got[side][name])
            summary[side] = (mean, se)
            print(f'{name if side == "seen" else "":<22} {side:<7} '
                  + ' '.join(f'{m:>11.4f} +-{e:.4f}' for m, e in zip(mean[:3], se[:3]))
                  + f' {mean[3] * 100:>9.2f}% +-{se[3] * 100:.2f}'
                  + (f'   ({games:,} games, {decisions:,} decisions)' if side == 'seen' else ''))
        (ms, ss), (mu, su) = summary['seen'], summary['unseen']
        gap, gse = mu - ms, np.hypot(ss, su)
        print(f'{"":<22} {"gap":<7} '
              + ' '.join(f'{g:>+11.4f} +-{e:.4f}' for g, e in zip(gap[:3], gse[:3]))
              + f' {gap[3] * 100:>+9.2f}% +-{gse[3] * 100:.2f}')
        print(f'{"":<22} {"":7} '
              + ' '.join(f'{g / e if e else 0:>+11.1f} se{"":6}' for g, e in zip(gap, gse)))

    print()
    print('the gap is the generalisation gap, in standard errors that count games')
    print('rather than decisions. Read it with the trend across checkpoints:')
    print('  unseen still falling  -> undertrained, the cheapest gain is more steps')
    print('  unseen flat, gap wide -> the corpus is the limit, and more of it is the lever')
    print('  unseen flat, gap thin -> capacity is the limit, and a wider trunk is')


if __name__ == "__main__":
    main()
