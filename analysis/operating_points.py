from paths import ROOT, RESULTS, output_root, task_cache
import sys, json
import numpy as np, pandas as pd
OUT = output_root()
sys.path.insert(0, str(ROOT))
import experiment as C
S = C.runner()
g = C.grid()
names = S.CR.variant_names()
floor = float(np.min(np.diff(g)))
manifest = json.loads((RESULTS / 'tasks/reference/reference_manifest.json').read_text(encoding='utf-8-sig'))
cache = task_cache(manifest)


records = {'dev': [], 'test': []}
for meta_path in sorted(cache.glob('*.npz.json')):
    meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
    role = meta['role']
    if role not in records or not meta['kinds'] or (role == 'dev' and meta['M'] not in [10, 100, 1000]):
        continue
    with np.load(str(meta_path)[:-5], allow_pickle=True) as z:
        for kind, km in meta['kinds'].items():
            if role == 'dev' and kind != 'base':
                continue
            d = {'domain': meta['domain'], 'structure': meta['structure'], 'kind': kind, 'M_base': int(meta['M']), 'seed': int(meta['seed']), 'M': int(km['M']), 'Z_sup': np.asarray(z[f'{kind}|Z_sup'], float), '_sorted_normal': np.sort(np.asarray(z[f'{kind}|Z_eval'], float))}
            if role == 'test':
                zp = np.asarray(z[f'{kind}|Z_pos'], float)
                pv = np.asarray(z[f'{kind}|pos_variant']).astype(str)
                d['_sorted_positive'] = {name: np.sort(zp[pv == name]) for name in names}
            records[role].append(d)
assert len(records['dev']) == 568 and len(records['test']) == 1016

fitted = json.loads((RESULTS / 'support_baselines/fitted_values.json').read_text(encoding='utf-8-sig'))
Z = pd.read_parquet(RESULTS / 'support_baselines/rows_new_baselines.parquet')
Z = Z[Z.method == 'Z-norm + NP rank']
key = lambda d: (d['domain'], d['structure'], d['kind'], d['M_base'], d['seed'])
conditions = list(dict.fromkeys(((d['domain'], d['structure'], d['M_base']) for d in records['dev'])))
dev_groups = [np.array([j for j, d in enumerate(records['dev']) if (d['domain'], d['structure'], d['M_base']) == c]) for c in conditions]
mu = {}
sd = {}
for n in [10, 20, 50, 100]:
    for role, regs in records.items():
        mu[role, n] = np.array([d['Z_sup'][:n].mean() for d in regs])
        sd[role, n] = np.array([max(d['Z_sup'][:n].std(), floor) for d in regs])

def dev_risk(boundary, shift, alpha, metric):
    far = []
    for d, b in zip(records['dev'], boundary):
        tau = C.succ(float(b + shift), g)
        z = d['_sorted_normal']
        far.append((len(z) - np.searchsorted(z, tau, side='left')) / len(z) if np.isfinite(tau) else 0.0)
    far = np.array(far)
    v = far if metric == 'FAR' else (far > alpha + 1e-12).astype(float)
    return float(np.mean([v[ix].mean() for ix in dev_groups]))
all_rows = []
selection = []
repro = []
for n in [10, 20, 50, 100]:
    for alpha in [0.01, 0.05]:
        beta = int(round(100 * (1 - alpha)))
        z = Z[(Z.n == n) & np.isclose(Z.alpha, alpha)]
        zlook = {tuple((r[k] for k in C.TASK_KEY)): r for r in z.to_dict('records')}
        fc = next((x for x in fitted if x.get('method') == 'Z-norm + NP rank' and x.get('n') == n and (x.get('alpha') == alpha)))
        c = fc['c']
        configs = [('Z-norm + NP rank', -1, mu['dev', n] + c * sd['dev', n], mu['test', n] + c * sd['test', n])]
        for seed in C.SEEDS:
            obj, md, path = C.load_model(S, n, 'base_set', alpha, seed)
            P = pd.read_parquet(RESULTS / f'main/arms/n{n}_b{beta}_base_set_s{seed}_np_global.parquet')
            look = {tuple((r[k] for k in C.TASK_KEY)): r for r in P.to_dict('records')}
            lam = float(P.Lambda.iloc[0])
            qd = np.array([C.predict(S, obj, d, alpha, 'base_set', n) for d in records['dev']])
            bt = np.array([look[key(d)]['q_pred'] + lam for d in records['test']])
            configs.append(('Meta set + np_global', seed, qd + lam, bt))
        for method, seed, bd, bt in configs:
            source = zlook if seed == -1 else {tuple((r[k] for k in C.TASK_KEY)): r for r in pd.read_parquet(RESULTS / f'main/arms/n{n}_b{beta}_base_set_s{seed}_np_global.parquet').to_dict('records')}
            maxerr = 0.0
            for d, b in zip(records['test'], bt):
                rr = source[key(d)]
                tau = C.succ(float(b), g)
                ev = C.fast_eval(tau, d, alpha, names)
                assert np.isinf(tau) and np.isinf(rr['tau']) or abs(tau - rr['tau']) < 1e-12
                maxerr = max(maxerr, abs(ev['FAR'] - rr['FAR']), abs(ev['TPR23'] - rr['TPR23']))
            assert maxerr < 1e-12
            repro.append({'method': method, 'n': n, 'alpha': alpha, 'seed': seed, 'max_recalculation_error': maxerr})
            for target_metric in ['V']:
                target = 0.05
                lo = -2.0
                hi = 2.0
                assert dev_risk(bd, lo, alpha, target_metric) > target and dev_risk(bd, hi, alpha, target_metric) <= target
                for _ in range(80):
                    mid = (lo + hi) / 2
                    if dev_risk(bd, mid, alpha, target_metric) <= target:
                        hi = mid
                    else:
                        lo = mid
                shift = hi
                dr = dev_risk(bd, shift, alpha, target_metric)
                assert dr <= target + 1e-12
                selection.append({'method': method, 'training_seed': seed, 'n': n, 'alpha': alpha, 'development_constraint': target_metric, 'target': target, 'shift': shift, 'development_realized_risk': dr, 'development_conditions': len(dev_groups), 'development_registries': 568, 'selection_uses_test': False})
                for d, b in zip(records['test'], bt):
                    tau = C.succ(float(b + shift), g)
                    rr = C.fast_eval(tau, d, alpha, names)
                    original = zlook[key(d)]
                    rr.update(domain=d['domain'], structure=d['structure'], kind=d['kind'], M=d['M_base'], M_eff=d['M'], seed=d['seed'], n=n, alpha=alpha, method=method, training_seed=seed, development_constraint=target_metric, shift=shift, baseline_BA23=original['baseline_BA23'])
                    all_rows.append(rr)

R = pd.DataFrame(all_rows)
R.to_parquet(OUT / 'development_adjusted_rows.parquet', index=False)
keys = ['n', 'alpha', 'method', 'training_seed', 'development_constraint', 'domain', 'structure', 'kind', 'M']
G = R.groupby(keys).agg(V=('violation', 'mean'), FAR=('FAR', 'mean'), TPR23=('TPR23', 'mean'), BA23=('BA23', 'mean'), baseline_BA23=('baseline_BA23', 'mean'), sentinel_rate=('sentinel', 'mean')).reset_index()
G['meets'] = (G.V <= 0.05 + 1e-12) & (G.BA23 >= G.baseline_BA23 - 1e-12)
G.to_csv(OUT / 'development_adjusted_conditions.csv', index=False, encoding='utf-8-sig')
PS = G.groupby(['n', 'alpha', 'method', 'training_seed', 'development_constraint']).agg(V=('V', 'mean'), FAR=('FAR', 'mean'), TPR23=('TPR23', 'mean'), BA23=('BA23', 'mean'), passing=('meets', 'sum')).reset_index()
PS.to_csv(OUT / 'development_adjusted_by_seed.csv', index=False, encoding='utf-8-sig')
PS.groupby(['n', 'alpha', 'method', 'development_constraint'])[['V', 'FAR', 'TPR23', 'BA23', 'passing']].mean().reset_index().to_csv(OUT / 'development_adjusted_summary.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(selection).to_csv(OUT / 'development_selection.csv', index=False, encoding='utf-8-sig')
