from __future__ import annotations
import time
import numpy as np, pandas as pd
from scipy.stats import binom
import experiment as C
from experiments import train_evaluate as F2
N = 100
ALPHAS = tuple((round(k / 100, 2) for k in range(1, 100)))
SMOKE_ALPHAS = (0.01, 0.5, 0.99)
HASH_NAME = 'Hash convention (10 bits)'
REF_NAME = 'Evaluation-query reference'

def tag_of(a):
    return f'b{int(round(round(1 - a, 2) * 100)):02d}'

def parse_part(part, alphas):
    if part in ('all', 'aggregate'):
        return (part, list(alphas))
    i, k = (int(x) for x in part.split('/'))
    if not 1 <= i <= k:
        raise ValueError('--part must be all, aggregate or i/K with 1 <= i <= K')
    return ('part', [a for j, a in enumerate(alphas) if j % k == i - 1])

def arm_done(arms, token):
    return F2.arm_complete(arms, token)

def expected_tokens(alphas, modes, seeds):
    t = []
    for a in alphas:
        tag = tag_of(a)
        t += [f'baseline_n{N}_{tag}', f'reference_n{N}_{tag}']
        t += [f'n{N}_{tag}_{m}_s{s}_{c}' for m in modes for s in seeds for c in F2.CAL]
    return t

def run_level(S, MF, a, test, fit, cal, dev, fixed, modes, seeds, arms, p):
    beta = round(1 - a, 2)
    tag = tag_of(a)
    n = N
    F2.STATE['N'] = n
    token = f'baseline_n{n}_{tag}'
    if not arm_done(arms, token):
        bf = S.FC.beta_fraction(a)
        rank = (bf.numerator * (n + 1) + bf.denominator - 1) // bf.denominator
        valid = [j for j in range(1, n + 1) if binom.sf(j - 1, n, beta) <= 0.05]
        k = min(valid) if valid else n + 1
        rows = []
        tau_h = MF.fit_C0h()
        for b in test:
            for kind, d in b.reg.items():
                z = np.sort(d['Z_sup'][:n])
                taus = dict(fixed)
                taus[HASH_NAME] = tau_h
                taus['Direct split CP'] = S.SIM.strict_successor(float(z[rank - 1])) if rank <= n else np.inf
                taus['Direct NP'] = S.SIM.strict_successor(float(z[k - 1])) if k <= n else np.inf
                for name, tau in taus.items():
                    rows.append(F2.score(S, b, kind, d, tau, n, a, name))
        F2.save_arm(pd.DataFrame(rows), token, arms)
        C.atomic_json(arms / f'{token}.metadata.json', dict(n=n, beta=beta, alpha=a, boundary_case=False, cp_rank=rank, np_rank=k, hash_tau=tau_h))

    token = f'reference_n{n}_{tag}'
    if not arm_done(arms, token):
        rows = [F2.score(S, b, kind, d, S.SIM.tau_empirical(d['Z_eval'], a), n, a, REF_NAME) for b in test for kind, d in b.reg.items()]
        F2.save_arm(pd.DataFrame(rows), token, arms)
        C.atomic_json(arms / f'{token}.metadata.json', dict(n=n, beta=beta, alpha=a, rule='similarity.tau_empirical(Z_eval, alpha)'))

    for mode in modes:
        for s in seeds:
            tok = f'n{n}_{tag}_{mode}_s{s}'
            if all(arm_done(arms, f'{tok}_{c}') for c in F2.CAL):


                continue
            t0 = time.perf_counter()
            obj, md, path = S.model(mode, C.VAR[mode], a, s, fit, dev)
            if md['epochs'] != 15 or md['pool'] != 'mean' or md.get('reused'):
                raise ValueError('stored model settings differ (epochs, pool or reused)')
            tt = time.perf_counter() - t0

            for c in F2.CAL:
                arm = f'{tok}_{c}'
                if arm_done(arms, arm):

                    continue
                corr = S.calibration(obj, mode, a, cal, c)
                lam = corr['global']['Lambda']
                rows = []
                for b in test:
                    for kind, d in b.reg.items():
                        q = S.predict(obj, d, a, mode, b.domain)
                        tau = S.SIM.strict_successor(S.FC.corrected_boundary(q, lam))
                        rows.append(F2.score(S, b, kind, d, tau, n, a, f'{C.METHODS[mode]} + {c}', s, q, lam))
                df = pd.DataFrame(rows)
                if len(df) != 1016:
                    raise ValueError('expected 1016 registries')
                F2.save_arm(df, arm, arms)
                C.atomic_json(arms / f'{arm}.metadata.json', dict(n=n, beta=beta, alpha=a, mode=mode, variant=C.VAR[mode], training_seed=s, model_path=str(path), model_sha256=C.sha256(path / 'model.pt'), metadata=md, calibration=corr, timing={'training_s': tt}))


def aggregate(out, arms, alphas, modes, seeds):
    missing = [t for t in expected_tokens(alphas, modes, seeds) if not arm_done(arms, t)]
    if missing:
        raise RuntimeError(f'Missing numerical results: {missing}')
    files = [arms / f'{token}.conditions.csv' for token in sorted(expected_tokens(alphas, modes, seeds))]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.assign(training_seed=d.training_seed.fillna(-1))
    ref = d[d.method == REF_NAME]
    d = d[d.method != REF_NAME]
    C.save_csv(d, out / 'conditions.csv')
    C.save_csv(ref, out / 'reference_conditions.csv')
    summ = C.method_summary(d)
    C.save_csv(summ, out / 'summary.csv')
    C.save_csv(C.method_summary(ref), out / 'reference_summary.csv')
    pool = d.groupby(['method', 'alpha', 'domain', 'kind', 'training_seed']).agg(FAR=('FAR', 'mean'), V=('V', 'mean'), TPR23=('TPR23', 'mean')).reset_index()
    pool = pool.groupby(['method', 'alpha', 'domain', 'kind'])[['FAR', 'V', 'TPR23']].mean().reset_index()
    C.save_csv(pool, out / 'pool_summary.csv')
    return summ

def main(args, out):
    t_start = time.time()
    S = C.runner()
    S.encode = F2.encode(S)
    from methods import fixed as MF
    alphas = SMOKE_ALPHAS if args.smoke else ALPHAS
    modes = F2.MODES
    seeds = C.SEEDS[:1] if args.smoke else C.SEEDS
    what, todo = parse_part(args.part, alphas)
    arms = C.output_path(out / 'arms')
    arms.mkdir(parents=True, exist_ok=True)
    mroot = C.OUT_ROOT / 'models_coverage_smoke' if args.smoke else C.ROOT / 'models_coverage'

    if what != 'aggregate':
        fit, cal, dev, test, fixed = F2.prepare(S, None, out)

        S.CACHE = C.output_path(mroot / f'n{N}')
        for a in todo:
            run_level(S, MF, a, test, fit, cal, dev, fixed, modes, seeds, arms, None)
    result = {'part': args.part, 'levels_run': len(todo) if what != 'aggregate' else 0}
    if what != 'part':
        summ = aggregate(out, arms, alphas, modes, seeds)

        prop = summ[summ.method == 'Meta set + np_global'].sort_values('alpha')
        result['proposed_mean_FAR_below_alpha_levels'] = int((prop.FAR <= prop.alpha + 1e-12).sum())
        result['proposed_V_le_delta_levels'] = int((prop.V <= C.DELTA + 1e-12).sum())
        result['levels'] = int(prop.alpha.nunique())
    result['elapsed_s'] = round(time.time() - t_start, 1)
    return result
if __name__ == '__main__':
    C.run_step('coverage', '99 nominal levels (n = 100)', main)
