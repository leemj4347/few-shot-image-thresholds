# Baselines for small support sets
from __future__ import annotations
import math
import numpy as np, pandas as pd
from scipy.stats import binom, genpareto
import experiment as C

NS = (10, 20, 50, 100)
ALPHAS = (0.01, 0.05)
U_Q = 0.90
K_MIN = 5
MODE = "base_set"


# Smallest k with P(Bin(N, 1-alpha) >= k) <= delta (Tong et al
def np_rank(N, alpha, delta=C.DELTA):
    beta = 1 - alpha
    k = int(binom.ppf(1 - delta, N, beta)) + 1
    k = max(1, k - 3)
    while k <= N and binom.sf(k - 1, N, beta) > delta: k += 1
    return k if k <= N else None


def np_threshold(scores, alpha, g):
    z = np.sort(np.asarray(scores, float))
    k = np_rank(len(z), alpha)
    return (C.succ(float(z[k - 1]), g), k) if k else (float("inf"), None)


def gpd_boundary(z, alpha):
    z = np.sort(np.asarray(z, float))
    n = len(z)
    u = float(np.quantile(z, U_Q, method="inverted_cdf"))
    y = z[z > u] - u
    k = len(y)
    if k < K_MIN: return float("inf"), "too_few_exceedances"
    p_u = k / n
    if alpha >= p_u: return u, "alpha_above_tail"
    try:
        xi, loc, sig = genpareto.fit(y, floc=0)
    except Exception:
        return float("inf"), "fit_failed"
    if not (np.isfinite(xi) and np.isfinite(sig) and sig > 0): return float("inf"), "fit_failed"
    r = alpha / p_u
    q = u + sig * math.log(1 / r) if abs(xi) < 1e-9 else u + sig / xi * (r ** (-xi) - 1)
    return float(q), "ok"


def main(args, out):
    ns = (10, 100) if args.smoke else NS
    seeds = C.SEEDS[:1] if args.smoke else C.SEEDS
    units = {"tasks": 1, "pooled": len(ALPHAS), "per_n": len(ns) * len(ALPHAS) * 2, "crc": len(ns) * len(ALPHAS) * len(seeds),
             "aggregate": 1}

    S, m, cfg, fit, cal, dev, test = C.load_reference(None)
    names = S.CR.variant_names()
    g = C.grid()
    C.prepare_test(test, S.SIM.grid_ceil(.63075), names)
    regs = [(b, k, d) for b in test for k, d in b.reg.items()]
    if args.smoke: regs = regs[:80]

    floor = float(np.min(np.diff(g)))
    rows, info = [], []

    fitM = sorted({int(b.reg["base"]["M"]) for b in fit})
    pooled = {}
    for a in ALPHAS:
        allz = np.concatenate([b.reg["base"]["Z_eval"] for b in fit])
        pooled[(a, "global")] = np_threshold(allz, a, g)
        for M in fitM:
            zM = np.concatenate([b.reg["base"]["Z_eval"] for b in fit if int(b.reg["base"]["M"]) == M])
            pooled[(a, M)] = np_threshold(zM, a, g)
        info.append(dict(method="Pooled NP", alpha=a, global_tau=pooled[(a, "global")][0], global_rank=pooled[(a, "global")][1],
                         pooled_scores=int(len(allz)), by_M={str(M): pooled[(a, M)][0] for M in fitM}))

    near = lambda M: min(fitM, key=lambda x: (abs(math.log10(x) - math.log10(M)), -x))
    for n in ns:
        for a in ALPHAS:
            for b, k, d in regs:
                rows.append(C.row(b, k, d, pooled[(a, "global")][0], n, a, "Pooled NP (global)", names))
                rows.append(C.row(b, k, d, pooled[(a, near(int(d["M"])))][0], n, a, "Pooled NP (nearest M)", names,
                                  pooled_M=near(int(d["M"]))))
                bd, status = gpd_boundary(np.asarray(d["Z_sup"], float)[:n], a)
                rows.append(C.row(b, k, d, C.succ(bd, g), n, a, "GPD tail (support)", names, fit_status=status))


            t = []
            for b in cal:
                d = b.reg["base"]
                z = np.asarray(d["Z_sup"], float)[:n]
                mu = z.mean()
                sd = max(z.std(), floor)
                t.append((C.target_quantile(S, d["Z_eval"], a) - mu) / sd)
            t = np.sort(t)
            kk = np_rank(len(t), C.DELTA)
            c = float(t[kk - 1]) if kk else float("inf")
            info.append(dict(method="Z-norm + NP rank", n=n, alpha=a, c=c, rank=kk, L=len(t)))
            for b, k, d in regs:
                z = np.asarray(d["Z_sup"], float)[:n]
                mu = z.mean()
                sd = max(z.std(), floor)
                rows.append(C.row(b, k, d, C.succ(mu + c * sd, g) if np.isfinite(c) else float("inf"), n, a,
                                  "Z-norm + NP rank", names))


            for s in seeds:
                obj, md, path = C.load_model(S, n, MODE, a, s)
                cr = []
                for b in cal:
                    d = b.reg["base"]
                    cr.append((C.predict(S, obj, d, a, MODE, n), np.sort(np.asarray(d["Z_eval"], float))))
                L = len(cr)

                def risk(lam):
                    tot = 0.0
                    for q, zz in cr:
                        tau = C.succ(q + lam, g)
                        tot += 0.0 if not np.isfinite(tau) else (len(zz) - np.searchsorted(zz, tau, side="left")) / len(zz)
                    return L / (L + 1) * tot / L + 1 / (L + 1)
                lo, hi = -2.0, 2.0
                if risk(hi) > a: lam = float("inf")
                else:
                    for _ in range(80):
                        mid = (lo + hi) / 2
                        if risk(mid) <= a: hi = mid
                        else: lo = mid
                    lam = hi
                info.append(dict(method="CRC same predictor", n=n, alpha=a, training_seed=s, Lambda=lam,
                                 calibrated_bound=risk(lam) if np.isfinite(lam) else None))
                A = pd.read_parquet(C.arm_path(n, round(1 - a, 2), MODE, s, "np_global"))
                look = {kk: q for kk, q in zip(A[C.TASK_KEY].itertuples(index=False, name=None), A.q_pred)}
                for b, k, d in regs:
                    q = look[(b.domain, b.structure, k, int(b.M), int(b.seed))]
                    rows.append(C.row(b, k, d, C.succ(q + lam, g) if np.isfinite(lam) else float("inf"), n, a,
                                      "CRC same predictor", names, seed=s, q_pred=q, Lambda=lam))

    R = pd.DataFrame(rows)
    C.save_parquet(R, out / "rows_new_baselines.parquet")
    C.atomic_json(out / "fitted_values.json", info)

    keep = [
        "Meta set + np_global",
        "Meta set + fisch_global",
        "Direct NP",
        "Direct split CP",
        "DBpHash fixed",
        "Youden J",
    ]
    old = []
    for n in ns:
        for a in ALPHAS:
            A = C.read_arms(n, a, seeds=seeds)
            old.append(A[A.method.isin(keep)])
    H = C.hash_convention_rows(args.smoke)
    if H is not None: old.append(H[H.n.isin(ns) & H.alpha.isin(ALPHAS)])
    O = pd.concat(old, ignore_index=True)
    if args.smoke:
        keys = {(b.domain, b.structure, k, int(b.M), int(b.seed)) for b, k, d in regs}
        O = O[[t in keys for t in O[C.TASK_KEY].itertuples(index=False, name=None)]]
    cond = C.condition_table(pd.concat([R, O], ignore_index=True))
    C.save_csv(cond, out / "conditions.csv")
    summ = C.method_summary(cond)
    C.save_csv(summ, out / "summary.csv")
    gs = R[R.method == "GPD tail (support)"].groupby(["n", "alpha"]).fit_status.value_counts().unstack(fill_value=0)
    gs.to_csv(C.output_path(out / "gpd_fit_status.csv"), encoding="utf-8-sig")

    return {"methods_new": sorted(R.method.unique()), "n": list(ns), "alphas": list(ALPHAS)}


if __name__ == "__main__":
    C.run_step("support_baselines", "적은 서포트용 기준선", main)
