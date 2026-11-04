from paths import RESULTS, output_root, task_cache
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
OUT = output_root()
OUT.mkdir(exist_ok=True)
B = 2000
RNG = 20261016
SEEDS = [42, 43, 44, 45, 46]
KEY = ['domain', 'structure', 'kind', 'M', 'seed']
COND = KEY[:-1]
protocol = {'bootstrap_draws': B, 'rng_seed': RNG, 'alphas': [0.01, 0.05], 'n': [10, 20, 50, 100], 'methods': ['Meta set + np_global', 'Z-norm + NP rank'], 'metrics': ['V', 'FAR', 'TPR23', 'BA23', 'passing'], 'resampling': ['within_condition', 'shared_bundle'], 'selection_role': 'dev', 'bisection_iterations': 80}


def source_sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
sourcepaths = [RESULTS / 'main/summary.csv', RESULTS / 'support_baselines/summary.csv', RESULTS / 'support_baselines/rows_new_baselines.parquet']
sourcehash = {str(p): source_sha(p) for p in sourcepaths}
if (RESULTS / 'task_flow.csv').is_file():
    flow = pd.read_csv(RESULTS / 'task_flow.csv')
    assert flow.constructed.sum() == 1628 and flow.excluded.sum() == 748
    flow.to_csv(OUT / 'task_flow.csv', index=False, encoding='utf-8-sig')
else:
    m = json.loads((RESULTS / 'tasks/reference/reference_manifest.json').read_text(encoding='utf-8-sig'))
    cache = task_cache(m)
    invalid = pd.DataFrame(json.loads((RESULTS / 'tasks/reference/invalid_tasks.json').read_text(encoding='utf-8-sig')))
    invalid['reason_group'] = invalid.reason.map(lambda x: 'insufficient_registry_pool' if x.startswith('registry:') else 'insufficient_normal_queries' if x.startswith('queries') else 'other')
    valid = []
    for p in sorted(cache.glob('*.complete.json')):
        rec = json.loads(p.read_text(encoding='utf-8-sig'))
        k = rec['key']
        valid.append(dict(role=k[2], domain=k[3], structure=k[4], M=k[5], seed=k[6], scheme=k[1], stem=p.name.removesuffix('.complete.json')))
    valid = pd.DataFrame(valid)
    invalid_keys = set(invalid[['role', 'domain', 'structure', 'M', 'seed']].itertuples(index=False, name=None))
    valid = valid[[x not in invalid_keys for x in valid[['role', 'domain', 'structure', 'M', 'seed']].itertuples(index=False, name=None)]]
    assert len(valid) == 1628 and len(invalid) == 748
    invalid.to_csv(OUT / 'excluded_tasks_all.csv', index=False, encoding='utf-8-sig')
    invalid.groupby(['role', 'domain', 'structure', 'M', 'reason_group'], dropna=False).size().rename('excluded').reset_index().to_csv(OUT / 'excluded_reasons.csv', index=False, encoding='utf-8-sig')
    a = valid.groupby(['role', 'domain', 'structure', 'M']).size().rename('constructed')
    b = invalid.groupby(['role', 'domain', 'structure', 'M']).size().rename('excluded')
    flow = pd.concat([a, b], axis=1).fillna(0).astype(int).reset_index()
    flow['planned'] = flow.constructed + flow.excluded
    flow['exclusion_rate'] = flow.excluded / flow.planned
    flow.to_csv(OUT / 'task_flow.csv', index=False, encoding='utf-8-sig')
    assert flow.planned.sum() == 2376
maincond = pd.read_csv(RESULTS / 'main/conditions.csv')
zcond = pd.read_csv(RESULTS / 'support_baselines/conditions.csv')
zcond = zcond[zcond.method == 'Z-norm + NP rank']
cond = pd.concat([maincond[maincond.method == 'Meta set + np_global'], zcond], ignore_index=True)
gkeys = ['method', 'n', 'alpha'] + COND
cm = cond.groupby(gkeys).agg(V=('V', 'mean'), FAR=('FAR', 'mean'), TPR23=('TPR23', 'mean'), n_tasks=('n_tasks', 'first')).reset_index()
cm.to_csv(OUT / 'condition_risks.csv', index=False, encoding='utf-8-sig')
worst = cm.sort_values('V').groupby(['method', 'n', 'alpha']).tail(1)
worst.to_csv(OUT / 'worst_conditions.csv', index=False, encoding='utf-8-sig')
Z = pd.read_parquet(RESULTS / 'support_baselines/rows_new_baselines.parquet')
Z = Z[Z.method == 'Z-norm + NP rank']
ci = []
diff = []
pointrows = []
metrics = ['V', 'FAR', 'TPR23', 'BA23', 'passing']
for n in [10, 20, 50, 100]:
    for alpha in [0.01, 0.05]:
        beta = round((1 - alpha) * 100)
        parts = []
        for seed in SEEDS:
            p = RESULTS / f'main/arms/n{n}_b{beta}_base_set_s{seed}_np_global.parquet'
            sourcehash[str(p)] = source_sha(p)
            parts.append(pd.read_parquet(p))
        P = pd.concat(parts, ignore_index=True)
        z = Z[(Z.n == n) & np.isclose(Z.alpha, alpha)].copy()
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
        for method, vals, src in [('Meta set + np_global', pp, 'main'), ('Z-norm + NP rank', zz, 'support_baselines')]:
            s = pd.read_csv(RESULTS / f'{src}/summary.csv')
            s = s[(s.method == method) & (s.n == n) & np.isclose(s.alpha, alpha)].iloc[0]
            for j, k in enumerate(metrics):
                assert abs(vals[j] - s[k]) < 1e-12, (n, alpha, method, k, vals[j], s[k])
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

pd.DataFrame(pointrows).to_csv(OUT / 'point_estimates.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(ci).to_csv(OUT / 'conditional_intervals.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(diff).to_csv(OUT / 'paired_differences.csv', index=False, encoding='utf-8-sig')
assert all((source_sha(Path(p)) == h for p, h in sourcehash.items()))
