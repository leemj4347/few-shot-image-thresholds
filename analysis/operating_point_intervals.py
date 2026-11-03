from paths import ROOT, RESULTS, output_root, task_cache
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
OUT = output_root()
B = 2000
RNG = 20261016
SEEDS = [42, 43, 44, 45, 46]
KEY = ['domain', 'structure', 'kind', 'M', 'seed']
COND = KEY[:-1]
metrics = ['V', 'FAR', 'TPR23', 'BA23', 'passing']
ci = []
diff = []
pointrows = []


R = pd.read_parquet(OUT / 'development_adjusted_rows.parquet')
selection = OUT / 'development_selection.csv'
before = hashlib.sha256(selection.read_bytes()).hexdigest()
for n in [10, 20, 50, 100]:
    for alpha in [0.01, 0.05]:
        use = R[(R.n == n) & np.isclose(R.alpha, alpha) & (R.development_constraint == 'V')]
        P = use[use.method == 'Meta set + np_global'].copy()
        z = use[use.method == 'Z-norm + NP rank'].copy()
        order = P[KEY].drop_duplicates().sort_values(KEY).reset_index(drop=True)
        order['pos'] = np.arange(len(order))
        assert len(order) == 1016
        p = P.merge(order, on=KEY)
        z = z.merge(order, on=KEY)
        assert len(z) == 1016 and len(p) == 5080
        cols = ['violation', 'FAR', 'TPR23', 'BA23']
        pm = np.stack([p[p.training_seed == s].sort_values('pos')[cols].astype(float).to_numpy() for s in SEEDS])
        zm = z.sort_values('pos')[cols].astype(float).to_numpy()[None]
        base = z.sort_values('pos').baseline_BA23.to_numpy()
        groups = {tuple(c): d.pos.to_numpy() for c, d in order.groupby(COND)}
        assert len(groups) == 15

        def stats(arr, indices=None, six=None):
            vals = []
            passes = []
            for c, ix0 in groups.items():
                item = None if indices is None else indices[c]
                if isinstance(item, tuple):
                    weights = item[1]
                    den = weights.sum(1)
                    assert (den > 0).all()
                    v = np.einsum('stj,bt->sbj', arr[:, ix0, :], weights) / den[None, :, None]
                    bb = weights @ base[ix0] / den
                else:
                    ix = ix0[None, :] if indices is None else ix0[item]
                    v = arr[:, ix, :].mean(2)
                    bb = base[ix].mean(1)
                v = v.transpose(1, 0, 2)
                if six is not None:
                    v = np.take_along_axis(v, six[:, :, None] if len(arr) > 1 else np.zeros_like(six)[:, :, None], axis=1)
                passes.append(((v[:, :, 0] <= 0.05 + 1e-12) & (v[:, :, 3] >= bb[:, None] - 1e-12)).astype(float))
                vals.append(v)
            means = np.mean(vals, axis=0).mean(1)
            pc = np.sum(passes, axis=0).mean(1)
            return np.column_stack([means, pc])
        pp = stats(pm)[0]
        zz = stats(zm)[0]
        for method, vals in [('Meta set + np_global', pp), ('Z-norm + NP rank', zz)]:
            pointrows.append(dict(n=n, alpha=alpha, method=method, **dict(zip(metrics, vals))))
        shared = {}
        for c, ix in groups.items():
            shared.setdefault((c[0], c[1], c[3]), []).append((c, ix))
        for scheme in ['within_condition', 'shared_bundle']:
            rng = np.random.default_rng(RNG)
            drawp = []
            drawz = []
            for start in range(0, B, 50):
                size = min(50, B - start)
                six = rng.integers(0, 5, (size, 5))
                ixs = {}
                if scheme == 'within_condition':
                    ixs = {c: rng.integers(0, len(ix), (size, len(ix))) for c, ix in groups.items()}
                else:
                    for strata, items in shared.items():
                        union = np.unique(np.concatenate([order.iloc[ix].seed.to_numpy() for c, ix in items]))
                        counts = rng.multinomial(len(union), np.full(len(union), 1 / len(union)), size=size)
                        for c, ix in items:
                            weights = counts[:, np.searchsorted(union, order.iloc[ix].seed.to_numpy())]
                            ixs[c] = ('weights', weights)
                drawp.append(stats(pm, ixs, six))
                drawz.append(stats(zm, ixs, six))
            dp = np.concatenate(drawp)
            dz = np.concatenate(drawz)
            for method, point, draw in [('Meta set + np_global', pp, dp), ('Z-norm + NP rank', zz, dz)]:
                for j, k in enumerate(metrics):
                    lo, hi = np.percentile(draw[:, j], [2.5, 97.5])
                    ci.append(dict(n=n, alpha=alpha, scheme=scheme, method=method, metric=k, estimate=point[j], ci_low=lo, ci_high=hi, draws=B))
            for j, k in enumerate(metrics):
                lo, hi = np.percentile(dp[:, j] - dz[:, j], [2.5, 97.5])
                diff.append(dict(n=n, alpha=alpha, scheme=scheme, metric=k, estimate=pp[j] - zz[j], ci_low=lo, ci_high=hi, draws=B))

pd.DataFrame(pointrows).to_csv(OUT / 'development_fixed_point_estimates.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(ci).to_csv(OUT / 'development_fixed_intervals.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(diff).to_csv(OUT / 'development_fixed_paired_differences.csv', index=False, encoding='utf-8-sig')
expected = pd.read_csv(OUT / 'development_adjusted_summary.csv')
got = pd.DataFrame(pointrows)
errors = []
for row in got.to_dict('records'):
    ref = expected[(expected.n == row['n']) & np.isclose(expected.alpha, row['alpha']) & (expected.method == row['method']) & (expected.development_constraint == 'V')].iloc[0]
    for k in metrics:
        errors.append(abs(row[k] - ref[k]))
assert max(errors) < 1e-12
assert before == hashlib.sha256(selection.read_bytes()).hexdigest()
