# Bootstrap intervals and paired tests
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import experiment as C

ALPHAS = (0.01, 0.05)
NS = (100, 10)
PROPOSED = "Meta set + np_global"
COMPARE = ["Meta set + fisch_global", "Summary H256 + np_global", "Hash convention (10 bits)", "Youden J",
           "Equal error rate", "Otsu", "DBpHash fixed", "Direct NP", "Direct split CP"]
METRICS = ["violation", "FAR", "TPR23", "BA23"]


def load(n, a, smoke):
    A = C.read_arms(n, a, modes=("base_set", "base_summary256"))
    H = C.hash_convention_rows(smoke)
    if H is not None: A = pd.concat([A, H[(H.n == n) & (H.alpha == a)]], ignore_index=True)

    keys = None
    for _, g in A.groupby("method"):
        k = set(g[C.TASK_KEY].itertuples(index=False, name=None))
        keys = k if keys is None else keys & k
    mask = [t in keys for t in A[C.TASK_KEY].itertuples(index=False, name=None)]
    return A[mask].copy()


# method -> metric -> (seeds, tasks) aligned on one task order; condition -> task positions
def arrays(A):
    A = A.copy()
    A["cond"] = list(zip(A.domain, A.structure, A.kind, A.M))
    order = A[["cond", "seed"]].drop_duplicates().sort_values(["cond", "seed"]).reset_index(drop=True)
    order["pos"] = np.arange(len(order))
    A = A.merge(order, on=["cond", "seed"])
    conds = {c: g.pos.to_numpy() for c, g in order.groupby("cond")}
    base = A.drop_duplicates("pos").sort_values("pos").baseline_BA23.to_numpy()
    out = {}
    for meth, g in A.groupby("method"):
        seeds = sorted(g.training_seed.unique())
        mats = {k: np.full((len(seeds), len(order)), np.nan) for k in METRICS + ["sentinel"]}
        for i, s in enumerate(seeds):
            h = g[g.training_seed == s]
            for k in mats: mats[k][i, h.pos.to_numpy()] = h[k].astype(float).to_numpy()
        if any(np.isnan(v).any() for v in mats.values()): raise ValueError(f"incomplete task coverage: {meth}")
        out[meth] = mats
    return out, conds, base, A


# Per seed: condition means, pass count; then mean over seeds
def stats_from(mats, base, conds, task_idx=None, seed_idx=None):
    S = mats["violation"].shape[0]
    res = {k: [] for k in ["V", "FAR", "TPR23", "BA23", "passing"]}
    pass_count = None
    for c, pos in conds.items():
        ix = pos[task_idx[c]] if task_idx is not None else pos[None, :]
        v = mats["violation"][:, ix].mean(-1)
        f = mats["FAR"][:, ix].mean(-1)
        t = mats["TPR23"][:, ix].mean(-1)
        ba = mats["BA23"][:, ix].mean(-1)
        bb = base[ix].mean(-1)
        if seed_idx is not None:
            sel = seed_idx if S > 1 else np.zeros_like(seed_idx)
            pick = lambda x: np.take_along_axis(x.T, sel, axis=1)
            v, f, t, ba = pick(v), pick(f), pick(t), pick(ba)
        else:
            v, f, t, ba = v.T, f.T, t.T, ba.T
        ok = (v <= C.DELTA + 1e-12) & (ba >= bb[:, None] - 1e-12)
        pass_count = ok.astype(float) if pass_count is None else pass_count + ok
        for k, x in zip(["V", "FAR", "TPR23", "BA23"], [v, f, t, ba]): res[k].append(x)
    out = {k: np.mean(np.stack(res[k]), axis=0).mean(-1) for k in ["V", "FAR", "TPR23", "BA23"]}
    out["passing"] = pass_count.mean(-1)
    return out


def main(args, out):
    B = 200 if args.smoke else args.bootstrap
    ns = (100,) if args.smoke else NS

    rng = np.random.default_rng(20261016)
    ci_rows, diff_rows, test_rows = [], [], []
    for n in ns:
        for a in ALPHAS:
            mats, conds, base, A = arrays(load(n, a, args.smoke))

            point = {m: stats_from(x, base, conds) for m, x in mats.items()}

            draws = {m: {k: np.empty(B) for k in ["V", "FAR", "TPR23", "BA23", "passing"]} for m in mats}
            chunk = 50
            for b0 in range(0, B, chunk):
                bb = min(chunk, B - b0)
                tix = {c: rng.integers(0, len(pos), size=(bb, len(pos))) for c, pos in conds.items()}
                six = rng.integers(0, len(C.SEEDS), size=(bb, len(C.SEEDS)))
                for m, x in mats.items():
                    r = stats_from(x, base, conds, tix, six)
                    for k in r: draws[m][k][b0:b0 + bb] = r[k]

            for m in mats:
                for k in ["V", "FAR", "TPR23", "BA23", "passing"]:
                    lo, hi = np.percentile(draws[m][k], [2.5, 97.5])
                    ci_rows.append(dict(n=n, alpha=a, method=m, metric=k, estimate=float(point[m][k][0]),
                                        ci_low=float(lo), ci_high=float(hi), draws=B))
            if PROPOSED in mats:
                for cmp in COMPARE:
                    if cmp not in mats: continue
                    for k in ["V", "TPR23", "passing"]:
                        dd = draws[PROPOSED][k] - draws[cmp][k]
                        lo, hi = np.percentile(dd, [2.5, 97.5])
                        diff_rows.append(dict(n=n, alpha=a, comparison=f"{PROPOSED} - {cmp}", metric=k,
                                              estimate=float(point[PROPOSED][k][0] - point[cmp][k][0]),
                                              ci_low=float(lo), ci_high=float(hi), draws=B))

            if PROPOSED in mats:
                for cmp in COMPARE:
                    if cmp not in mats: continue
                    for k in ["violation", "TPR23"]:
                        x = mats[PROPOSED][k].mean(0)
                        y = mats[cmp][k].mean(0)
                        d = x - y
                        nz = int(np.count_nonzero(np.abs(d) > 1e-15))
                        pv = float(wilcoxon(x, y, zero_method="wilcox").pvalue) if nz >= 10 else np.nan
                        test_rows.append(dict(n=n, alpha=a, comparison=f"{PROPOSED} vs {cmp}", metric=k,
                                              mean_difference=float(d.mean()), nonzero_pairs=nz, registries=len(d),
                                              test="Wilcoxon signed-rank, two-sided", p_value=pv))

    T = pd.DataFrame(test_rows)
    if len(T):
        ok = T.p_value.notna()
        m = int(ok.sum())
        order = T[ok].p_value.sort_values().index
        adj = pd.Series(np.nan, index=T.index)
        run = 0.0
        for i, idx in enumerate(order):
            run = max(run, min(1.0, (m - i) * T.loc[idx, "p_value"]))
            adj[idx] = run
        T["p_holm"] = adj
    C.save_csv(pd.DataFrame(ci_rows), out / "C1_bootstrap_ci.csv")
    C.save_csv(pd.DataFrame(diff_rows), out / "C1_paired_differences_ci.csv")
    C.save_csv(T, out / "C3_paired_tests.csv")
    return {"bootstrap_draws": B, "ci_rows": len(ci_rows), "tests": len(test_rows)}


if __name__ == "__main__":
    C.run_step("statistics", "Bootstrap intervals and paired tests", main)
