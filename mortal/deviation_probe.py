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
the decisions it picked. For one fixed setting that average is unbiased --
no row's prediction ever saw its own label. Choosing the setting is a
separate matter: reading every alpha's out-of-sample score and reporting
the best one has read the labels being averaged, and is optimistic by
however wide the scan was. So every alpha's gain is printed and none is
called the answer. A probe that has found real headroom is positive down
the column; one that is fitting noise sits at zero, which is what the
current policy already gets.

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


def prepare(train, test, data, rank):
    """The part of a fold that does not depend on how hard it is regularised.

    The basis comes from the training fold only -- fitting it on everything
    would let the test rows shape their own features -- and it costs an SVD
    of the whole fold, so it is built once and every alpha reuses it.
    """
    phi = data['phi']
    gap = np.log(np.maximum(data['p_forced'], 1e-12)) - np.log(data['p_argmax'])
    mean, sd = phi[train].mean(0), phi[train].std(0) + 1e-6
    basis = np.linalg.svd((phi[train] - mean) / sd, full_matrices=False)[2][:rank].T
    build = lambda m: design(phi[m], data['argmax'][m], data['forced'][m], gap[m],
                             basis, mean, sd, rank)
    xtr, xte = build(train), build(test)
    # And the normal equations, which do not depend on alpha either.
    return xtr.T @ xtr, xtr.T @ data['label'][train], xte


def fit_and_predict(prepared, alpha, free=2):
    gram, rhs, xte = prepared
    pen = np.full(gram.shape[0], alpha)
    pen[:free] = 0.
    return xte @ np.linalg.solve(gram + np.diag(pen), rhs)


def overlap(a_tr, f_tr, a_te, f_te):
    """How much two decisions' action pairs have in common.

    The interaction block puts `+z` in the forced action's slot and `-z` in
    the argmax's, so the inner product between two rows of it is the states'
    inner product times this: +1 for each end that agrees, -1 for each end
    that agrees with the other's opposite. Writing it out means the kernel
    never has to build the 47,104 columns it stands for.
    """
    return ((f_te[:, None] == f_tr[None, :]).astype(np.int8)
            + (a_te[:, None] == a_tr[None, :])
            - (f_te[:, None] == a_tr[None, :])
            - (a_te[:, None] == f_tr[None, :]))


def full_probe(data, parts, alphas):
    """The same probe with every trunk dimension, not the leading few.

    Principal components are ordered by variance, and nothing says the
    direction that carries "this action is better than the head thinks" is a
    high-variance one. In the dual form the rank cap disappears: the design
    is [state | state x action pair] over all 1024 dimensions, and its Gram
    matrix is the states' Gram times one plus their action overlap.

    The intercept and the logit gap stay unpenalised, exactly as in the
    primal version, and that is not the same thing as fitting them first and
    handing the kernel the leftovers -- the trunk features and the logit gap
    come from the same trunk and are not orthogonal, so a two-step fit
    solves a different problem. Since there are only two free columns the
    exact joint solution is cheap: with `A = K + alpha I`, stationarity in
    the free block gives `(X' A^-1 X) b = X' A^-1 y`, a 2x2 system, and the
    dual weights are then `A^-1 (y - X b)`.
    """
    phi, y = data['phi'], data['label']
    gap = np.log(np.maximum(data['p_forced'], 1e-12)) - np.log(data['p_argmax'])
    am, fo = data['argmax'], data['forced']
    n = len(y)
    out = {a: np.zeros(n) for a in alphas}
    for test in parts:
        train = ~test
        mean, sd = phi[train].mean(0), phi[train].std(0) + 1e-6
        z = ((phi - mean) / sd).astype(np.float64)
        x = np.column_stack([np.ones(n), gap])
        ktr = (z[train] @ z[train].T) * (1 + overlap(am[train], fo[train],
                                                     am[train], fo[train]))
        kte = (z[test] @ z[train].T) * (1 + overlap(am[train], fo[train],
                                                    am[test], fo[test]))
        xtr, ytr = x[train], y[train].astype(np.float64)
        for alpha in alphas:
            a = ktr + alpha * np.eye(len(ktr))
            solved = np.linalg.solve(a, np.column_stack([xtr, ytr]))
            ainv_x, ainv_y = solved[:, :xtr.shape[1]], solved[:, -1]
            beta = np.linalg.solve(xtr.T @ ainv_x, xtr.T @ ainv_y)
            c = ainv_y - ainv_x @ beta
            out[alpha][test] = x[test] @ beta + kte @ c
        del ktr, kte, z
    return out


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
    ap.add_argument('--alpha', type=float, nargs='+', default=[1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--full', action='store_true',
                    help='every trunk dimension, in the dual form, instead of the '
                         'leading --rank principal components. Components are ordered '
                         'by variance and the direction that matters need not be a '
                         'high-variance one, so this is the version with no such gap')
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

    # The one number the policy already exposes, fitted the same way, is what
    # the probe has to beat. Infinite regularisation turns the probe into
    # exactly this, so it is the benchmark and the floor at once -- which is
    # why the alpha grid has to run high enough to approach it.
    base = np.zeros(n)
    for test in parts:
        x = np.column_stack([np.ones((~test).sum()), gap[~test]])
        b = np.linalg.lstsq(x, y[~test], rcond=None)[0]
        base[test] = np.column_stack([np.ones(test.sum()), gap[test]]) @ b
    bench = 1 - ((y - base) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f'\nthe logit gap alone, out of sample: R^2 {bench:+.5f}')

    where = 'every dimension' if args.full else f'rank {args.rank}'
    print(f'\n=== out of sample, {len(parts)} folds split by block, {where} ===')
    print(f"{'alpha':>10} {'R^2 vs mean':>12} {'R^2 vs logit gap':>17} {'corr':>7}")
    # Every alpha's out-of-fold prediction, solved once and read by both tables
    # below: one n-vector an alpha, where solving again for the second table
    # doubled the work and grew with --rank.
    if args.full:
        every = full_probe(data, parts, args.alpha)
    else:
        ready = [prepare(~t, t, data, args.rank) for t in parts]
        every = {}
        for alpha in args.alpha:
            every[alpha] = np.zeros(n)
            for test, got in zip(parts, ready):
                every[alpha][test] = fit_and_predict(got, alpha)
    for alpha in args.alpha:
        pred = every[alpha]
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2b = 1 - ((y - pred) ** 2).sum() / ((y - base) ** 2).sum()
        c = np.corrcoef(pred, y)[0, 1]
        print(f'{alpha:>10.0f} {r2:>12.5f} {r2b:>17.5f} {c:>7.4f}')

    print()
    print('=== what following the probe would have been worth, at every alpha ===')
    print("For one fixed alpha the average is unbiased: no row's prediction ever saw")
    print('its own label. Picking the best row of this table is not -- that choice')
    print('reads the labels being averaged, and is optimistic by however wide the')
    print('scan was. So no row here is "the" answer: a probe that found real headroom')
    print('is positive down the column, not in one cell of it.')
    print(f'{"alpha":>10} {"threshold":>10} {"switched":>9} {"share":>7} '
          f'{"gain per decision, GRP":>27}')
    for alpha in args.alpha:
        for t, k, share, per, se in value_of_following(every[alpha], y):
            flag = '' if se == 0 else f'  ({per / se:+.1f} se)'
            print(f'{alpha:>10.0f} {t:>10.2f} {k:>9,} {share:>6.1f}% '
                  f'{per:>+16.5f} +-{se:.5f}{flag}')

    print('\nfor comparison, the same selection rule on the labels themselves '
          '-- the ceiling no probe can pass')
    take = y > 0
    print(f'  switch wherever the deviation actually helped: {y[take].sum() / n:+.5f} '
          f'over {take.mean() * 100:.0f}% of decisions')
    print('  that number is not achievable; it is what perfect hindsight is worth, '
          'and it is the scale the column above should be read against')


if __name__ == '__main__':
    main()
