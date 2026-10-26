# Hamming-distance convention versus the FAR budget
from __future__ import annotations
import numpy as np, pandas as pd
import experiment as C
ALPHAS = (0.01, 0.05, 0.2)
NS = (100, 10)
FAR_LEVELS = np.r_[0.0, np.logspace(-4, 0, 41)]


# FAR and TPR23 at every grid value for one registry
def curves(d, g, names):
    zz = d["_sorted_normal"]
    far = 1.0 - np.searchsorted(zz, g, side="left") / len(zz)
    tpr = np.zeros(len(g))
    for x in names:
        v = d["_sorted_positive"][x]
        tpr += 1.0 - np.searchsorted(v, g, side="left") / len(v)
    return (far, tpr / len(names))


def main(args, out, holder):
    S, MF, T, REG = C.modules()
    names = S.CR.variant_names()
    alphas = (0.01, 0.05) if args.smoke else ALPHAS
    ns = (100,) if args.smoke else NS
    p = C.Progress(
        "hash_thresholds",
        "Hamming 관행값과 FAR 예산",
        {"tasks": 1, "hash_rule": len(alphas), "reference": len(alphas), "methods": len(alphas) * len(ns), "curves": 1},
        args.smoke,
        args.log,
    )
    holder["progress"] = p
    S, m, cfg, fit, cal, dev, test = C.load_reference(p)
    p.advance("tasks")
    g = C.grid()
    tau_dbp = S.SIM.grid_ceil(0.63075)
    C.prepare_test(test, tau_dbp, names)
    regs = [(b, k, d) for b in test for k, d in b.reg.items()]
    if args.smoke:
        regs = regs[:60]
    tau_h = MF.fit_C0h()
    C.atomic_json(out / "hash_rule.json", {"tau_grid": tau_h})
    hrows = []
    for a in alphas:
        for b, k, d in regs:
            for n in ns:
                hrows.append(C.row(b, k, d, tau_h, n, a, "Hash convention (10 bits)", names))
        p.advance("hash_rule", alpha=a)
    H = pd.DataFrame(hrows)
    C.save_parquet(H, out / "hash_convention_rows.parquet")
    hc = C.condition_table(H)
    C.save_csv(hc, out / "hash_convention_conditions.csv")
    ref = []
    for a in alphas:
        for b, k, d in regs:
            t = S.SIM.tau_empirical(d["Z_eval"], a)
            r = C.fast_eval(t, d, a, names)
            ref.append(dict(
                domain=b.domain,
                structure=b.structure,
                kind=k,
                M=int(b.M),
                seed=int(b.seed),
                alpha=a,
                tau_eval_min=t,
                FAR_ref=r["FAR"],
                TPR_ref=r["TPR23"],
            ))
        p.advance("reference", alpha=a)
    R = pd.DataFrame(ref)
    C.save_csv(R, out / "reference_point.csv")
    allrows = []
    for n in ns:
        for a in alphas:
            A = C.read_arms(n, a, modes=("base_set", "base_summary256"))
            A = pd.concat([A, H[(H.n == n) & (H.alpha == a)]], ignore_index=True)
            A = A.merge(R[R.alpha == a].drop(columns="alpha"), on=C.TASK_KEY, how="inner")
            A["FAR_over_alpha"] = A.FAR / a
            A["TPR_gap"] = np.where(A.violation, np.nan, A.TPR_ref - A.TPR23)
            allrows.append(A)
            p.advance("methods", n=n, alpha=a)
    A = pd.concat(allrows, ignore_index=True)
    keep = C.K + [
        "seed",
        "tau",
        "FAR",
        "violation",
        "TPR23",
        "BA23",
        "baseline_BA23",
        "sentinel",
        "tau_eval_min",
        "FAR_ref",
        "TPR_ref",
        "FAR_over_alpha",
        "TPR_gap",
    ]
    C.save_parquet(A[keep], out / "per_task_budget.parquet")
    cg = A.groupby(C.K, dropna=False).agg(FAR=("FAR", "mean"), V=("violation", "mean"), TPR23=("TPR23", "mean"), TPR_ref=("TPR_ref", "mean"), FAR_over_alpha=("FAR_over_alpha", "mean"), TPR_gap_nonviol=("TPR_gap", "mean")).reset_index()
    met = ["FAR", "V", "TPR23", "TPR_ref", "FAR_over_alpha", "TPR_gap_nonviol"]
    s = cg.groupby(["method", "n", "alpha", "training_seed"])[met].mean().reset_index()
    s = s.groupby(["method", "n", "alpha"])[met].mean().reset_index()
    C.save_csv(s, out / "summary_budget.csv")
    cond = C.condition_table(A.drop(columns=["tau_eval_min", "FAR_ref", "TPR_ref", "FAR_over_alpha", "TPR_gap"]))
    C.save_csv(C.method_summary(cond), out / "summary_pass_counts.csv")
    p.stage("curves")
    fr = []
    for b, k, d in regs:
        far, tpr = curves(d, g, names)
        best = [float(tpr[far <= f].max()) if np.any(far <= f) else 0.0 for f in FAR_LEVELS]
        fr.append(dict(domain=b.domain, kind=k, M=int(b.M), **{f"f{i}": v for i, v in enumerate(best)}))
    F = pd.DataFrame(fr)
    cols = [f"f{i}" for i in range(len(FAR_LEVELS))]
    front = F.groupby(["domain", "kind", "M"])[cols].mean().mean()
    curve = pd.DataFrame({"FAR_level": FAR_LEVELS, "TPR_frontier_mean": front.values})
    C.save_csv(curve, out / "frontier_curve.csv")
    pts = s[s.n == 100][["method", "alpha", "FAR", "TPR23", "V"]]
    C.save_csv(pts, out / "operating_points_n100.csv")
    p.advance("curves")
    hs = hc[hc.n == 100].groupby("alpha").meets.sum().to_dict()
    return {
        "hash_convention_tau": tau_h,
        "hash_convention_passing_by_alpha": {str(k): int(v) for k, v in hs.items()},
        "registries": len(regs),
    }
if __name__ == "__main__":
    C.run_step("hash_thresholds", "Hamming 관행값과 FAR 예산", main)
