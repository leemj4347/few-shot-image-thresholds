# Baselines for small support sets
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import binom
import experiment as C

NS = (10, 20, 50, 100)
ALPHAS = (0.01, 0.05)
MODE = "base_set"


# Smallest k with P(Bin(N, 1-alpha) >= k) <= delta (Tong et al
def np_rank(N, alpha, delta=C.DELTA):
    beta = 1 - alpha
    k = int(binom.ppf(1 - delta, N, beta)) + 1
    k = max(1, k - 3)
    while k <= N and binom.sf(k - 1, N, beta) > delta: k += 1
    return k if k <= N else None






def main(args, out):
    ns = (10, 100) if args.smoke else NS
    seeds = C.SEEDS[:1] if args.smoke else C.SEEDS
    S, m, cfg, fit, cal, dev, test = C.load_reference(None)
    names = S.CR.variant_names()
    g = C.grid()
    C.prepare_test(test, S.SIM.grid_ceil(.63075), names)
    regs = [(b, k, d) for b in test for k, d in b.reg.items()]
    if args.smoke: regs = regs[:80]

    floor = float(np.min(np.diff(g)))
    rows, info = [], []

    for n in ns:
        for a in ALPHAS:
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

    return {"methods_new": sorted(R.method.unique()), "n": list(ns), "alphas": list(ALPHAS)}


if __name__ == "__main__":
    C.run_step("support_baselines", "적은 서포트용 기준선", main)
