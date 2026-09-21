"""Does the frozen trunk know something the linear policy head does not?

Three runs of paired replay established that the policy's action ordering is
right wherever it was tested, and that its confidence tracks the size of the
gap. What none of them asked is whether the 1024 trunk features hold
ordering information the single linear layer on top of them fails to read.
That is the difference between "the model is not smart enough" and "there is
nothing left to learn here", and it is the last cheap question in the
programme: the states are on disk, and no games have to be played.

The usual way to ask -- fit something and look at R^2 -- is weak here. The
kyoku's return has a standard deviation near 0.9 and the effects are 0.03 to
0.23, so almost all the variance is irreducible and R^2 is a small number
divided by a small number. The policy's own logit gap, which demonstrably
carries real information, manages 0.003.

So the question is put the way it would be used instead. A probe is trained
on part of the data and asked, on the part it has never seen, which
deviations it thinks are improvements. Following that advice means playing
the forced action at those decisions and the argmax everywhere else, and the
value of doing so is just the mean of the measured causal advantages over
the decisions it picked -- an unbiased estimate, because the picking used no
information from the labels being averaged. A probe that has found real
headroom comes out positive. A probe that is fitting noise comes out at
zero, which is what the current policy already gets.

    python deviation_probe.py /root/eval/deviation/*/features.npz
"""
import argparse

import numpy as np

# The head is one linear map of the trunk, so a probe that is allowed a
# different linear map of the same features and still finds nothing has
# answered the capacity question for linear readers. The interaction below
# gives it more than that: a separate reading per action.
DEFAULT_RANK = 32


def load(paths):
    """Every measured deviation, with the argmax nulls left out.

    Forcing the argmax over itself has a label of exactly zero by
    construction. Those rows check the apparatus; they carry no signal and
    would flatter any probe that learned to predict zero.
    """
    keep = {}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        names = [str(x) for x in d['rule_names']]
        live = np.array([names[i] != 'argmax' for i in d['rule']])
        for k in ('phi', 'label', 'argmax', 'forced', 'p_argmax', 'p_forced',
                  'legal', 'pass_legal', 'block', 'forced_rank'):
            keep.setdefault(k, []).append(d[k][live])
        keep.setdefault('run', []).append(np.full(int(live.sum()), len(keep['run'])
                                                  if 'run' in keep else 0))
    return {k: np.concatenate(v) for k, v in keep.items()}


def folds(group, k, rng):
    """Split by block, so no two rows of one block land on both sides.

    The rows are separate hanchans and would be safe to split at random.
    Splitting by block costs nothing and also rules out anything the blocks
    share -- a drifting machine, a batch effect in the arena -- being read as
    signal the probe found.

    A run with fewer blocks than folds cannot be split that way, and one
    fold is not a held-out set at all, so that falls back to splitting rows.
    Only a smoke test should ever take that branch.
    """
    blocks = np.unique(group)
    if len(blocks) >= k:
        rng.shuffle(blocks)
        return [np.isin(group, part) for part in np.array_split(blocks, k)]
    order = rng.permutation(len(group))
    return [np.isin(np.arange(len(group)), part)
            for part in np.array_split(order, k)]


def design(phi, argmax, forced, gap, basis, mean, scale, rank):
    """What the probe reads: the state, and the state per action compared.

    `z` is the state in the coordinates the training fold found most of its
    variance in. The interaction block holds `+z` in the forced action's slot
    and `-z` in the argmax's, so the probe can learn that some direction of
    the representation means "this action is better than the head thinks"
    rather than only "deviating is expensive around here".
    """
    z = ((phi - mean) / scale) @ basis
    n, k = z.shape
    inter = np.zeros((n, 46 * k), dtype=np.float32)
    rows = np.arange(n)
    for a, sign in ((forced, 1.), (argmax, -1.)):
        for j in range(k):
            inter[rows, a * k + j] += sign * z[:, j]
    return np.column_stack([np.ones(n, np.float32), gap, z, inter])


def ridge(x, y, alpha, free=2):
    """Least squares with everything but the first `free` columns shrunk."""
    pen = np.full(x.shape[1], alpha)
    pen[:free] = 0.
    return np.linalg.solve(x.T @ x + np.diag(pen), x.T @ y)


def fit_and_predict(train, test, data, rank, alpha):
    phi = data['phi']
    gap = np.log(np.maximum(data['p_forced'], 1e-12)) - np.log(data['p_argmax'])
    mean, sd = phi[train].mean(0), phi[train].std(0) + 1e-6
    # The basis comes from the training fold only; fitting it on everything
    # would let the test rows shape their own features.
    u, s, vt = np.linalg.svd((phi[train] - mean) / sd, full_matrices=False)
    basis = vt[:rank].T
    build = lambda m: design(phi[m], data['argmax'][m], data['forced'][m], gap[m],
                             basis, mean, sd, rank)
    beta = ridge(build(train), data['label'][train], alpha)
    return build(test) @ beta


def value_of_following(pred, label, thresholds=(0., .02, .05, .1)):
    """What playing the probe's suggestions would have been worth per decision.

    `pred` was produced without seeing any of these labels, so averaging the
    labels it selected is unbiased. The comparison is against the policy as
    it stands, which takes the argmax everywhere and scores exactly zero on
    this scale by construction.
    """
    out = []
    for t in thresholds:
        take = pred > t
        if take.sum() < 2:
            out.append((t, int(take.sum()), 0., 0., 0.))
            continue
        got = label[take]
        # Per decision offered, not per decision switched: switching nothing
        # is always available, so the rate has to be part of the number.
        per = got.sum() / len(label)
        se = got.std(ddof=1) * np.sqrt(len(got)) / len(label)
        out.append((t, int(take.sum()), take.mean() * 100, per, se))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('features', nargs='+')
    ap.add_argument('--rank', type=int, default=DEFAULT_RANK)
    ap.add_argument('--alpha', type=float, nargs='+', default=[30., 100., 300., 1000., 3000.])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    data = load(args.features)
    y = data['label']
    n = len(y)
    gap = np.log(np.maximum(data['p_forced'], 1e-12)) - np.log(data['p_argmax'])
    print(f'{n:,} measured deviations, trunk features {data["phi"].shape[1]}, '
          f'label sd {y.std(ddof=1):.3f}')
    print(f'as it stands, the policy takes the argmax at every one of them, '
          f'which is 0.0000 on this scale by definition')

    rng = np.random.default_rng(args.seed)
    parts = folds(data['block'], args.folds, rng)

    print(f'\n=== out of sample, {len(parts)} folds split by block ===')
    print(f"{'alpha':>8} {'R^2 vs mean':>12} {'R^2 vs logit gap':>17} {'corr':>7}")
    best, best_pred = None, None
    for alpha in args.alpha:
        pred = np.zeros(n)
        for test in parts:
            pred[test] = fit_and_predict(~test, test, data, args.rank, alpha)
        # The two things worth beating: predicting the mean, and predicting
        # from the one number the policy already exposes.
        base = np.zeros(n)
        for test in parts:
            x = np.column_stack([np.ones((~test).sum()), gap[~test]])
            b = np.linalg.lstsq(x, y[~test], rcond=None)[0]
            base[test] = np.column_stack([np.ones(test.sum()), gap[test]]) @ b
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2b = 1 - ((y - pred) ** 2).sum() / ((y - base) ** 2).sum()
        c = np.corrcoef(pred, y)[0, 1]
        print(f'{alpha:>8.0f} {r2:>12.5f} {r2b:>17.5f} {c:>7.4f}')
        if best is None or r2 > best[1]:
            best, best_pred = (alpha, r2), pred.copy()

    print(f'\n=== what following the best probe (alpha={best[0]:.0f}) would have been worth ===')
    print(f'{"threshold":>10} {"switched":>9} {"share":>7} '
          f'{"gain per decision, GRP":>27}')
    for t, k, share, per, se in value_of_following(best_pred, y):
        flag = '' if se == 0 else f'  ({per / se:+.1f} se)'
        print(f'{t:>10.2f} {k:>9,} {share:>6.1f}% {per:>+16.5f} +-{se:.5f}{flag}')

    print('\nfor comparison, the same selection rule on the labels themselves '
          '-- the ceiling no probe can pass')
    take = y > 0
    print(f'  switch wherever the deviation actually helped: {y[take].sum() / n:+.5f} '
          f'over {take.mean() * 100:.0f}% of decisions')
    print('  that number is not achievable; it is what perfect hindsight is worth, '
          'and it is the scale the column above should be read against')


if __name__ == '__main__':
    main()
