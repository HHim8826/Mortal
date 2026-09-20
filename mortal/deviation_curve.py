"""What one forced deviation cost, read against how far down the policy ranked it.

`deviation_cost.py` writes the pairs; this reads them back. Its own report
prints the four rules and the pooled total, which is enough when the rules are
the question. It is not enough here, for three reasons that each get a section
below.

Rank is not comparable across states. The worst of three choices is rank 2 and
the worst of fourteen is rank 13, and those are not the same kind of decision:
a call -- pon, chi, pass -- offers a handful of options and usually matters,
an opening discard offers thirteen and usually does not. So the curve is drawn
three times: against raw rank, against rank as a fraction of the options there
were, and inside bands of comparable width.

A mean of zero has two very different explanations. The action may be worth
the same everywhere, or better in one identifiable kind of state and worse in
another. Only the second is worth training on, so the moderators are tested.

And the obvious way to test them is wrong. See `moderators`.

    python deviation_curve.py /root/eval/deviation/sweep/deviations.json
"""
import argparse
import json

import numpy as np

RULES = ('argmax', 'rank2', 'median', 'worst')


def diff(rows):
    """The measured quantity: what the deviation did to that kyoku's return."""
    return np.array([r['kyoku_fork'] - r['kyoku_base'] for r in rows])


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def effect(rows):
    if not rows:
        return 0., 0.
    x = diff(rows)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.


def paired(rows, a, b):
    x = col(rows, a) - col(rows, b)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.


def interval(rows, draws=4000, seed=7):
    x = diff(rows)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def by_rule(rows):
    print('cost of one deviation, by which action replaced the argmax')
    print(f"{'':>8} {'n':>5} {'rank':>5} {'legal':>6} {'p(forced)':>9} {'moved':>6} "
          f"{'GRP delta':>20} {'95% boot':>18} {'placement':>16} {'pt':>18}")
    for rule in RULES:
        rs = [r for r in rows if r['rule'] == rule]
        if not rs:
            continue
        m, se = effect(rs)
        lo, hi = interval(rs)
        pm, pse = paired(rs, 'rank_fork', 'rank_base')
        tm, tse = paired(rs, 'pt_fork', 'pt_base')
        moved = np.mean([r['forked_at'] is not None for r in rs]) * 100
        print(f'{rule:>8} {len(rs):>5} {col(rs, "forced_rank").mean():>5.1f} '
              f'{col(rs, "legal").mean():>6.1f} {col(rs, "p_forced").mean():>9.3f} '
              f'{moved:>5.0f}% {m:>+9.4f} +-{se:>7.4f} [{lo:>+7.4f},{hi:>+7.4f}] '
              f'{pm:>+7.4f} +-{pse:>6.4f} {tm:>+8.3f} +-{tse:>7.3f}')

    print('\ndifferences between the rules')
    have = {r: [x for x in rows if x['rule'] == r] for r in RULES}
    have = {k: v for k, v in have.items() if v}
    keys = list(have)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ma, sa = effect(have[a])
            mb, sb = effect(have[b])
            d, s = mb - ma, np.hypot(sa, sb)
            print(f'  {b:>7} - {a:<7} {d:>+8.4f} +-{s:.4f}  ({abs(d / s):>4.1f} se)'
                  if s else f'  {b:>7} - {a:<7} {d:>+8.4f}')

    nulls = have.get('argmax', [])
    bad = [r for r in nulls if r['kyoku_fork'] != r['kyoku_base']]
    print(f'\nnull control: forcing the argmax over itself moved {len(bad)} of '
          f'{len(nulls)} rows'
          f'{"" if bad else " -- clean"}')
    if bad:
        print('  THE APPARATUS IS MEASURING SOMETHING IT SHOULD NOT')


def against_rank(rows):
    live = [r for r in rows if r['rule'] != 'argmax']

    print('\nagainst the actual rank of the forced action, rules pooled')
    print(f"{'rank':>9} {'n':>5} {'p(forced)':>10} {'GRP delta':>20} {'placement':>16}")
    edges = [1, 2, 3, 4, 6, 9, 13, 10 ** 6]
    for lo, hi in zip(edges, edges[1:]):
        rs = [r for r in live if lo <= r['forced_rank'] < hi]
        if len(rs) < 20:
            continue
        m, se = effect(rs)
        pm, pse = paired(rs, 'rank_fork', 'rank_base')
        label = f'{lo}' if hi == lo + 1 else f'{lo}-{hi - 1}' if hi < 10 ** 6 else f'{lo}+'
        print(f'{label:>9} {len(rs):>5} {col(rs, "p_forced").mean():>10.2e} '
              f'{m:>+9.4f} +-{se:>7.4f} {pm:>+7.4f} +-{pse:>6.4f}')

    print('\nagainst how far down, as a fraction of the choices there were')
    print(f"{'depth':>9} {'n':>5} {'rank':>5} {'legal':>6} {'GRP delta':>20}")
    for lo, hi in [(0., .15), (.15, .35), (.35, .65), (.65, .9), (.9, 1.01)]:
        rs = [r for r in live
              if r['legal'] > 1 and lo <= r['forced_rank'] / (r['legal'] - 1) < hi]
        if len(rs) < 20:
            continue
        m, se = effect(rs)
        print(f'{lo:.2f}-{hi:<4.2f} {len(rs):>5} {col(rs, "forced_rank").mean():>5.1f} '
              f'{col(rs, "legal").mean():>6.1f} {m:>+9.4f} +-{se:>7.4f}')

    print('\nthe rules again, inside bands of comparable width')
    bands = [('calls and few choices (legal<=5)', lambda r: r['legal'] <= 5),
             ('middling (6-9)', lambda r: 6 <= r['legal'] <= 9),
             ('wide open (10+)', lambda r: r['legal'] >= 10)]
    for label, keep in bands:
        part = [r for r in rows if keep(r)]
        if len(part) < 40:
            continue
        print(f'  {label}: n={len(part)}')
        for rule in RULES:
            rs = [r for r in part if r['rule'] == rule]
            if len(rs) < 20:
                continue
            m, se = effect(rs)
            print(f'    {rule:>7} n={len(rs):>5} rank {col(rs, "forced_rank").mean():>4.1f} '
                  f'p={col(rs, "p_forced").mean():.3f}  {m:>+8.4f} +-{se:.4f}')


def moderators(rows):
    """Where deviating pays, if anywhere -- and the split that looks like it does.

    Grouping the effect by the baseline arm's outcome manufactures a gradient.
    The effect is `fork - base`, so selecting on anything correlated with
    `base` puts `base` on both sides: the group whose base ran high has a fork
    that regresses toward the mean, and the difference comes out negative by
    construction. Under the null the two arms are exchangeable, so grouping by
    their average is fair and grouping by one of them is not. Both are printed
    because the gap between them is the point.
    """
    print('\n=== is the ordering ever wrong, or only wrong by nothing on average ===')
    print('\nthe same split done two ways, quartiles of the grouping variable')
    for rule in RULES[1:]:
        rs = [r for r in rows if r['rule'] == rule]
        if len(rs) < 100:
            continue
        b, f = col(rs, 'kyoku_base'), col(rs, 'kyoku_fork')
        y = f - b
        m, se = effect(rs)
        print(f'  {rule} (n={len(rs)}), overall {m:+.4f} +-{se:.4f}')
        for name, on in (('baseline arm only (biased)', b),
                         ('both arms averaged (fair)', (f + b) / 2)):
            q = np.percentile(on, [25, 50, 75])
            parts = [y[on <= q[0]], y[(on > q[0]) & (on <= q[1])],
                     y[(on > q[1]) & (on <= q[2])], y[on > q[2]]]
            print(f'    {name:<28} ' + '  '.join(f'{p.mean():+.3f}' for p in parts))

    print('\nleast squares on pre-treatment covariates only, second-choice rows')
    rs = [r for r in rows if r['rule'] == 'rank2']
    if len(rs) < 100:
        return
    feat = {
        'p(argmax)':       [r['p_argmax'] for r in rs],
        'legal actions':   [r['legal'] for r in rs],
        'kyoku':           [r['kyoku'] for r in rs],
        'through hanchan': [r['index'] / max(1, r['decisions']) for r in rs],
    }
    x = np.column_stack([np.ones(len(rs))] + [np.array(v, float) for v in feat.values()])
    y = diff(rs)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - x.shape[1]
    se = np.sqrt(np.diag((resid @ resid / dof) * np.linalg.pinv(x.T @ x)))
    for name, b, s in zip(['intercept'] + list(feat), beta, se):
        print(f'  {name:<18} {b:>+9.4f} +-{s:>7.4f}   ({b / s:>+5.1f} se)')
    f = ((y.var() * len(y) - resid @ resid) / (x.shape[1] - 1)) / (resid @ resid / dof)
    print(f'  joint F({x.shape[1] - 1},{dof}) = {f:.2f}  '
          f'(around 1 means none of this predicts anything)')

    print('\nthe distribution, not just its mean')
    for rule in RULES[1:]:
        rs = [r for r in rows if r['rule'] == rule]
        if not rs:
            continue
        v = diff(rs)
        moved = v[v != 0]
        if len(moved) < 2:
            continue
        print(f'  {rule:>7}: {len(moved) / len(v) * 100:>4.0f}% of pairs moved; among those '
              f'median {np.median(moved):>+7.4f}, {np.mean(moved > 0) * 100:>4.1f}% helped, '
              f'sd {moved.std(ddof=1):.3f}')


def precision(rows):
    """What pairing bought, and what it would take to see each gap."""
    print('\n=== what pairing bought ===')
    base = col(rows, 'kyoku_base')
    print(f'the kyoku GRP delta itself: mean {base.mean():+.4f}, sd {base.std(ddof=1):.4f}, '
          f'range [{base.min():.2f}, {base.max():.2f}]')
    print(f"{'':>8} {'paired sd':>10} {'unpaired':>9} {'ratio':>7} {'corr':>6} "
          f"{'n for 2 se':>11} {'unpaired':>9}")
    for rule in RULES[1:]:
        rs = [r for r in rows if r['rule'] == rule]
        if len(rs) < 100:
            continue
        x, y = col(rs, 'kyoku_base'), col(rs, 'kyoku_fork')
        sd = (y - x).std(ddof=1)
        loose = np.sqrt(x.var(ddof=1) + y.var(ddof=1))
        gap = abs((y - x).mean())
        need = int((2 * sd / gap) ** 2) if gap else 0
        need_loose = int((2 * loose / gap) ** 2) if gap else 0
        print(f'{rule:>8} {sd:>10.3f} {loose:>9.3f} {(loose / sd) ** 2:>6.1f}x '
              f'{np.corrcoef(x, y)[0, 1]:>6.3f} {need:>11,} {need_loose:>9,}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', nargs='?',
                    default='/root/eval/deviation/sweep/deviations.json')
    args = ap.parse_args()

    with open(args.results) as f:
        d = json.load(f)
    rows = d['rows']
    seen = d['identical'] + d['changed']
    print(f'{len(rows):,} forced deviations from {args.results}')
    print(f"untouched hanchans: {d['identical']:,} identical, {d['changed']} not "
          f"({d['changed'] / seen * 100:.2f}% moved); {d['rejected']} targets rejected, "
          f"{d['mismatched']} never came round\n")

    by_rule(rows)
    against_rank(rows)
    moderators(rows)
    precision(rows)


if __name__ == '__main__':
    main()
