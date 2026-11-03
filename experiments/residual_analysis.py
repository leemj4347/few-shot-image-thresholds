# Residual distribution and exchangeability check
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
import experiment as C

ALPHAS = (0.01, 0.05)
MODE = "base_set"
N = 100


def main(args, out):
    seeds = C.SEEDS[:1] if args.smoke else C.SEEDS

    S, m, cfg, fit, cal, dev, test = C.load_reference(None)

    cal_rows, test_rows, lam_rows = [], [], []
    for a in ALPHAS:
        arms = {s: pd.read_parquet(C.arm_path(N, round(1 - a, 2), MODE, s, "np_global")) for s in seeds}
        for s in seeds:
            obj, md, path = C.load_model(S, N, MODE, a, s)
            rows = []
            for b in cal:
                d = b.reg["base"]
                q = C.predict(S, obj, d, a, MODE, N)
                t = C.target_quantile(S, d["Z_eval"], a)
                rows.append((q, d["Z_eval"]))
                cal_rows.append(dict(alpha=a, training_seed=s, domain=b.domain, M=int(b.M), seed=int(b.seed),
                                     q_pred=q, q_target=t, residual=t - q))
            lam = C.residual_lambda(S, rows, a)
            stored = arms[s].Lambda.unique()
            lam_rows.append(dict(alpha=a, training_seed=s, Lambda=lam["Lambda"], rank=lam["rank"], L=lam["L"],
                                 stored_Lambda=float(stored[0]) if len(stored) == 1 else np.nan,
                                 match=bool(len(stored) == 1 and stored[0] == lam["Lambda"])))
            A = arms[s]
            if A.duplicated(C.TASK_KEY).any(): raise ValueError("stored result not unique for registry")
            look = {tuple(k): r for k, r in zip(A[C.TASK_KEY].itertuples(index=False, name=None), A.itertuples(index=False))}
            for b in test:
                for kind, d in b.reg.items():
                    t = C.target_quantile(S, d["Z_eval"], a)
                    h = look[(b.domain, b.structure, kind, int(b.M), int(b.seed))]
                    test_rows.append(dict(alpha=a, training_seed=s, domain=b.domain, structure=b.structure, kind=kind,
                                          M=int(b.M), seed=int(b.seed), q_pred=float(h.q_pred), q_target=t,
                                          residual=t - float(h.q_pred), Lambda=float(h.Lambda),
                                          violation=bool(h.violation), FAR=float(h.FAR)))

    CR = pd.DataFrame(cal_rows)
    TR = pd.DataFrame(test_rows)
    LR = pd.DataFrame(lam_rows)
    TR["exceeds_lambda"] = TR.residual > TR.Lambda
    C.save_parquet(CR, out / "residuals_calibration.parquet")
    C.save_parquet(TR, out / "residuals_test.parquet")
    C.save_csv(LR, out / "lambda_check.csv")
    rows = []
    for (a, dom, kind), g in TR.groupby(["alpha", "domain", "kind"]):
        ref = CR[CR.alpha == a].residual.to_numpy()
        ks = ks_2samp(g.residual.to_numpy(), ref)
        q = np.percentile(g.residual, [5, 50, 95])
        rows.append(dict(alpha=a, domain=dom, kind=kind, registries_x_seeds=len(g), residual_p05=q[0],
                         residual_median=q[1], residual_p95=q[2], share_exceeds_lambda=float(g.exceeds_lambda.mean()),
                         violation_rate=float(g.violation.mean()), ks_stat_vs_calibration=float(ks.statistic)))
    G = pd.DataFrame(rows)
    C.save_csv(G, out / "residual_by_domain_kind.csv")
    incl = TR[TR.exceeds_lambda == False]


    return {"lambda_matches_stored": bool(LR.match.all()),
            "inclusion_check_violations_when_r_le_lambda": int(incl.violation.sum())}


if __name__ == "__main__":
    C.run_step("residuals", "잔차 분포와 교환가능성 점검", main)
