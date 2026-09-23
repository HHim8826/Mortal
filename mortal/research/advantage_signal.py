"""Is a sampled alternative actually worse than the argmax, in the data?

    python -m research.advantage_signal --policy logs/policy/policy-t0.05.pth \
        --logs '/root/eval3s/dev/vs-*/<ident>/seeds-*' --games 4000

Phase 3 trained three configurations of PPO for 600,000 self-play games and
none of them changed the policy's strength. The positive control says where to
look: with the entropy bonus off and nothing anchoring, the policy did not
sharpen. A policy gradient sharpens only if the advantage it sees favours the
argmax.

An earlier version of this docstring said sharpening was worth +0.023 of rank
"by direct measurement". It was not: that number came from the
online560-offline520 comparison, which is a different pair of models. Phase 2
measured sampling at T=0.05 at -1.13 +- 1.53 pt against argmax -- not
distinguishable from zero. The question below is worth asking either way, and
its answer turned out to be yes.

So ask the data directly, with no training involved. Every decision in a
self-play log was either the policy's own best action or one of the
alternatives sampling drew instead. Both kinds are followed by the same thing:
the GRP's change in expected placement utility over that kyoku, which is the
return PPO was maximising. If deviating is worse, decisions that deviated are
followed by a smaller return.

The comparison has to be made within comparable states. Sampling deviates where
the policy is unsure, and those states are not the average state: a close
decision may sit in a kyoku that was going badly whether or not the tile was
right. Bucketing by the probability the policy gave its own best action keeps
like with like -- inside a bucket, deviating and not deviating happen in
states the policy rates equally difficult.

What the numbers mean:

    a clear negative gap      the signal is there and PPO should have found it;
                              the fault is in the trainer
    a gap indistinguishable   the data does not say the argmax is better at
    from zero                 the per-decision level, and no policy gradient on
                              this return can learn it, whatever the settings
"""
import argparse
import glob
import logging
from os import path

import numpy as np
import torch
from torch.utils.data import DataLoader

import prelude                                          # noqa: F401
from config import config
from dataloader import FileDatasetsIter, worker_init_fn
from model import Brain, DQN, KyokuValue, PolicyHead


def load_policy(file, device, value_from=None):
    state = torch.load(file, weights_only=True, map_location='cpu')
    cfg = state['config']
    brain = Brain(version=4, conv_channels=cfg['resnet']['conv_channels'],
                  num_blocks=cfg['resnet']['num_blocks']).eval()
    brain.load_state_dict(state['mortal'])
    head = PolicyHead(version=4).eval()
    head.load_state_dict(state['policy'])
    # The baseline the trainer subtracts: the run's own value head if it has
    # one, otherwise the teacher's dueling value stream, which is where every
    # run started.
    held = state if value_from is None else torch.load(
        value_from, weights_only=True, map_location='cpu')
    if 'value' in held:
        value = KyokuValue().eval()
        value.load_state_dict(held['value'])
    else:
        teacher = DQN(version=4)
        teacher.load_state_dict(held['current_dqn'])
        value = KyokuValue.from_dueling(teacher).eval()
    return (brain.to(device).requires_grad_(False),
            head.to(device).requires_grad_(False),
            value.to(device).requires_grad_(False))


def pooled(quantity, best_p, took_best, edges, rows=None):
    """The inverse-variance pooled gap across the buckets, on these rows."""
    if rows is not None:
        quantity, best_p, took_best = quantity[rows], best_p[rows], took_best[rows]
    total, weight = 0., 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (best_p >= lo) & (best_p < hi)
        a, b = quantity[inside & took_best], quantity[inside & ~took_best]
        if len(a) < 2 or len(b) < 2:
            continue
        var = a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)
        if not var > 0:
            continue
        total += (b.mean() - a.mean()) / var
        weight += 1 / var
    return total / weight if weight else np.nan


def groups_of(label):
    """The rows belonging to each cluster, as a list to draw from."""
    order = np.argsort(label, kind='stable')
    starts = np.searchsorted(label[order], np.arange(label.max() + 1))
    return [order[a:b] for a, b in zip(starts, np.append(starts[1:], len(order)))]


def bootstrap(quantity, label, best_p, took_best, edges, draws=400, seed=0):
    """The same estimator over resamples of whole clusters.

    Resampling decisions says a kyoku's thirty decisions are thirty pieces of
    evidence about its one outcome. Resampling kyoku says they are one, which
    is what they are.
    """
    groups = groups_of(label)
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for d in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        out[d] = pooled(quantity, best_p, took_best, edges,
                        np.concatenate([groups[i] for i in pick]))
    return out


def split(quantity, label):
    """The variance inside a cluster and between clusters, as components.

    Not the mean squares. The mean square between carries a cluster-size
    multiple of the between variance *plus* the within variance, so reading
    it as "how much is between" says half of pure noise is clustered. The
    one-way random-effects estimator takes the within part back out.

    `paid` must come back with nothing inside: gamma is 1 and every decision
    in a kyoku is paid the same GRP delta. Anything else means the labels are
    not the clusters they claim to be.
    """
    counts = np.bincount(label)
    counts = counts[counts > 0]
    _, inverse = np.unique(label, return_inverse=True)
    means = np.bincount(inverse, weights=quantity) / counts
    n, k = len(quantity), len(counts)
    within = float(((quantity - means[inverse]) ** 2).sum() / max(n - k, 1))
    msb = float((counts * (means - quantity.mean()) ** 2).sum() / max(k - 1, 1))
    # The cluster size the mean square between is scaled by, which is the
    # plain mean only when every cluster is the same size.
    m0 = (n - (counts ** 2).sum() / n) / max(k - 1, 1) if k > 1 else float(n)
    between = max(0., (msb - within) / m0) if m0 > 0 else 0.
    return within, between, n / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True, help='the policy that played these games')
    ap.add_argument('--value', default=None,
                    help='a checkpoint holding a trained value head')
    ap.add_argument('--logs', required=True, nargs='+',
                    help='directories or globs of <seed>_<key>_<seat>.json.gz')
    ap.add_argument('--player', default='mortal', help='the seat that sampled')
    ap.add_argument('--games', type=int, default=4000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--buckets', type=int, default=5)
    ap.add_argument('--draws', type=int, default=400, help='cluster bootstrap replicates')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--file-batch-size', type=int, default=4)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    device = torch.device(args.device or config['control']['device'])
    brain, head, value_head = load_policy(args.policy, device, args.value)

    files = []
    for pattern in args.logs:
        hits = sorted(glob.glob(pattern))
        for hit in hits:
            if path.isdir(hit):
                files.extend(sorted(glob.glob(path.join(hit, '*.json.gz'))))
            elif hit.endswith('.json.gz'):
                files.append(hit)
    files = files[:args.games]
    if not files:
        raise SystemExit(f'no game logs under {args.logs}')
    logging.info(f'{len(files):,} game logs, seat {args.player!r}')

    data = FileDatasetsIter(
        version=4, file_list=files, pts=list(config['env']['pts']),
        player_names=[args.player], file_batch_size=args.file_batch_size,
        num_epochs=1, final_rank=True, decision_ids=True,
    )
    batches = DataLoader(dataset=data, batch_size=args.batch_size, drop_last=False,
                         num_workers=args.workers, pin_memory=True,
                         worker_init_fn=worker_init_fn)

    best_p, took_best, paid, baseline, cluster = [], [], [], [], []
    with torch.inference_mode():
        for (obs, actions, masks, steps_to_done, kyoku_rewards, _ranks, _final,
             game_id, kyoku_id, _index) in batches:
            obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
            actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
            masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
            with torch.autocast(device.type):
                phi = brain(obs)
                logits = head(phi, masks).float()
                baseline.append(value_head(phi).float().cpu().numpy())
            p = logits.softmax(-1)
            best = logits.argmax(-1)
            best_p.append(p.gather(-1, best[:, None]).squeeze(-1).cpu().numpy())
            took_best.append((actions == best).cpu().numpy())
            # The same return the trainer used: gamma is 1 in every config this
            # has run under, so it is the kyoku's own GRP delta.
            paid.append((kyoku_rewards.double() *
                         config['env']['gamma'] ** steps_to_done.double()).numpy())
            # The kyoku a decision belongs to. Every decision in one shares a
            # GRP delta, which is what makes them one observation rather than
            # thirty.
            cluster.append(np.stack((game_id.numpy(), kyoku_id.numpy()), axis=1))
    del batches
    if data.iterator is not None:
        data.iterator.close()

    best_p = np.concatenate(best_p)
    took_best = np.concatenate(took_best)
    paid = np.concatenate(paid)
    baseline = np.concatenate(baseline)
    cluster = np.concatenate(cluster)
    # Contiguous labels to resample by: the kyoku, which is what shares a GRP
    # delta, and the trajectory it sits in, which is the wider unit a
    # robustness check uses.
    _, by_kyoku = np.unique(cluster, axis=0, return_inverse=True)
    _, by_game = np.unique(cluster[:, 0], return_inverse=True)
    n = len(paid)
    logging.info(f'{n:,} decisions, {1 - took_best.mean():.2%} of them not the argmax')

    # Only decisions the policy could have deviated on say anything: where one
    # action is legal there is nothing to choose and nothing to learn.
    edges = np.quantile(best_p, np.linspace(0, 1, args.buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    # The return itself, and the advantage the trainer actually multiplies into
    # the gradient. If the first has a gap and the second does not, the
    # baseline is eating the signal -- which is the question worth asking,
    # because the implementation that reports this working subtracts no
    # baseline at all.
    for name, quantity in (('paid', paid), ('paid - V(s)', paid - baseline)):
        print()
        print(name)
        print(f'{"P(best)":>16}  {"decisions":>10}  {"deviated":>8}  '
              f'{"argmax":>10}  {"other":>10}  {"gap":>18}')
        total, weight = 0., 0.
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = (best_p >= lo) & (best_p < hi)
            a, b = quantity[inside & took_best], quantity[inside & ~took_best]
            if len(a) < 2 or len(b) < 2:
                continue
            gap = b.mean() - a.mean()
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            total += gap / se ** 2
            weight += 1 / se ** 2
            print(f'{lo:>7.3f}-{hi:<8.3f} {inside.sum():>10,}  {len(b) / inside.sum():>7.1%}  '
                  f'{a.mean():>+10.4f}  {b.mean():>+10.4f}  {gap:>+10.4f} +-{se:.4f}')
        if weight:
            gap, flat = total / weight, weight ** -0.5
            print(f'{"pooled, decisions iid":>21}: {gap:+.4f} +- {flat:.4f} '
                  f'({gap / flat:+.1f} se) -- which they are not')
            for unit, label in (('kyoku', by_kyoku), ('game', by_game)):
                draws = bootstrap(quantity, label, best_p, took_best, edges, args.draws)
                se = float(np.nanstd(draws, ddof=1))
                lo, hi = np.nanpercentile(draws, [2.5, 97.5])
                print(f'{"resampling " + unit + "s":>21}: {gap:+.4f} +- {se:.4f} '
                      f'({gap / se:+.1f} se) [{lo:+.4f}, {hi:+.4f}], '
                      f'{se / flat:.1f}x the error above')

        within, between, m = split(quantity, by_kyoku)
        total = within + between
        share = within / total if total else float('nan')
        deff = 1 + (m - 1) * (1 - share)
        print(f'{"variance":>21}: {within:.4f} inside a kyoku, {between:.4f} between them '
              f'({100 * share:.1f}% inside, {m:.1f} decisions each)')
        print(f'{"so a batch of 256":>21}: carries {256 / deff:.0f} independent samples')

    print()
    print('negative means the sampled alternative was worth less than the argmax, '
          'which is the signal a policy gradient needs')
    print('`paid` must be exactly flat inside a kyoku -- gamma is 1 and every decision in '
          'one is paid the same GRP delta -- so anything but 0.0000 there means the '
          'clusters are wrong and nothing else here can be trusted')


if __name__ == '__main__':
    main()
