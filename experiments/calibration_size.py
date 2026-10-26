# Sensitivity to the number of calibration tasks
from __future__ import annotations
import numpy as np, pandas as pd
import experiment as C

ALPHAS = (0.01, 0.05)
SIZES = (59, 100, 150, 200, 250, 324)
MODE = "base_set"
N = 100


def main(args, out, holder):
    seeds = C.SEEDS[:1] if args.smoke else C.SEEDS
    R = 3 if args.smoke else args.repeats
    jobs = len(ALPHAS) * len(seeds) * sum(1 if L == 324 else R for L in SIZES)
    p = C.Progress(
        "calibration_size",
        "보정 태스크 수 민감도",
        {"tasks": 1, "subsets": jobs, "aggregate": 1},
        args.smoke,
        args.log,
    )
    holder["progress"] = p
    S, m, cfg, fit, cal, dev, test = C.load_reference(p)
    names = S.CR.variant_names()
    g = C.grid()
    C.prepare_test(test, S.SIM.grid_ceil(.63075), names)
    regs = [(b, k, d) for b in test for k, d in b.reg.items()]
    p.advance("tasks")
    rng = np.random.default_rng(20261016)
    rec = []
    for a in ALPHAS:
        for s in seeds:
            obj, md, path = C.load_model(S, N, MODE, a, s)
            rows = [(C.predict(S, obj, b.reg["base"], a, MODE, N), b.reg["base"]["Z_eval"]) for b in cal]
            A = pd.read_parquet(C.arm_path(N, round(1 - a, 2), MODE, s, "np_global"))
            look = {k: q for k, q in zip(A[C.TASK_KEY].itertuples(index=False, name=None), A.q_pred)}
            qs = [look[(b.domain, b.structure, k, int(b.M), int(b.seed))] for b, k, d in regs]
            stored = float(A.Lambda.iloc[0])
            for L in SIZES:
                for r in range(1 if L == 324 else R):
                    idx = np.arange(len(rows)) if L == 324 else np.sort(rng.choice(len(rows), L, replace=False))
                    lam = C.residual_lambda(S, [rows[i] for i in idx], a)["Lambda"]
                    res = []
                    for (b, k, d), q in zip(regs, qs):
                        tau = C.succ(S.FC.corrected_boundary(q, lam), g)
                        res.append(C.row(b, k, d, tau, N, a, f"L={L}", names, seed=s))
                    cond = C.condition_table(pd.DataFrame(res))
                    rec.append(dict(alpha=a, training_seed=s, L=L, repeat=r, Lambda=lam,
                                    full_matches_stored=(lam == stored) if L == 324 else None,
                                    V=cond.V.mean(), FAR=cond.FAR.mean(), TPR23=cond.TPR23.mean(), BA23=cond.BA23.mean(),
                                    passing=int(cond.meets.sum()), far_ok=int(cond.meets_far.sum()),
                                    sentinel_rate=cond.sentinel_rate.mean()))
                    p.advance("subsets", alpha=a, training_seed=s, L=L, repeat=r)
    D = pd.DataFrame(rec)
    C.save_csv(D, out / "calsize_runs.csv")
    summ = D.groupby(["alpha", "L"]).agg(runs=("V", "size"), Lambda_mean=("Lambda", "mean"),
                                         V_mean=("V", "mean"), V_p975=("V", lambda x: np.percentile(x, 97.5)),
                                         TPR_mean=("TPR23", "mean"), passing_mean=("passing", "mean"),
                                         far_ok_mean=("far_ok", "mean"), share_runs_V_le_delta=("V", lambda x: float(np.mean(x <= C.DELTA)))).reset_index()
    C.save_csv(summ, out / "calsize_summary.csv")
    p.advance("aggregate")
    full = D[D.L == 324]
    return {"full_L_matches_stored_lambda": bool(full.full_matches_stored.all()), "runs": len(D)}


if __name__ == "__main__":
    C.run_step("calibration_size", "보정 태스크 수 민감도", main)
