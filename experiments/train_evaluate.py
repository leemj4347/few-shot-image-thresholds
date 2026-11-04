# Train predictors and evaluate threshold methods
from __future__ import annotations
import time
import numpy as np, pandas as pd
from scipy.stats import binom
import experiment as C
MODES = ("base_set", "base_summary256")
NS = (10, 20, 50, 100)
BETAS = (0.99, 0.95, 0.8)
CAL = ("fisch_global", "np_global")
STATE = {"N": 100}


def encode(S):

    def enc(d, a, mode, domain):
        N = STATE["N"]
        z = np.asarray(d["Z_sup"], float)[:N]
        if len(z) != N or not np.isfinite(z).all():
            raise ValueError(f"support set needs {N} finite scores")
        c = np.log10(d["M"])
        return np.column_stack((z, np.full(N, c))) if mode.endswith("_set") else np.r_[S.F.SUMMARY.features(z, N, d["M"], a)[:7], c]
    return enc


def fast_eval(tau, d, a):
    sentinel = not np.isfinite(tau)
    zz = d["_sorted_normal"]
    far = 0.0 if sentinel else float((len(zz) - np.searchsorted(zz, tau, side="left")) / len(zz))
    r = {
        "tau": tau,
        "sentinel": bool(sentinel),
        "FAR": far,
        "violation": bool(far > a + 1e-12),
        "excess": far - a,
        "n_eval": len(zz),
        "n_pos_orig": int(d.get("n_pos_orig", 0)),
    }
    for name, v in d["_sorted_positive"].items():
        r["TPR_" + name] = (0.0 if sentinel else float((len(v) - np.searchsorted(v, tau, side="left")) / len(v))) if len(v) else np.nan
    return r


def score(S, b, kind, d, tau, n, a, name, seed=-1, q=None, lam=None):
    r = fast_eval(tau, d, a)
    vals = [r["TPR_" + x] for x in S.CR.variant_names()]
    if not all(np.isfinite(vals)):
        raise ValueError("TPR is undefined for at least one edit type (no positive rows)")
    r["TPR23"] = float(np.mean(vals))
    r["BA23"] = (r["TPR23"] + 1 - r["FAR"]) / 2
    r.update(
        domain=b.domain,
        structure=b.structure,
        kind=kind,
        M=int(b.M),
        M_eff=int(d["M"]),
        scheme="dbp",
        n=n,
        alpha=a,
        beta=round(1 - a, 2),
        method=name,
        training_seed=seed,
        seed=int(b.seed),
        q_pred=q,
        Lambda=lam,
        boundary_case=False,
        baseline_BA23=d["_baseline_BA23"],
    )
    return r


def arm_complete(arms, token):
    return all((arms / (token + suffix)).is_file() for suffix in (".parquet", ".conditions.csv", ".metadata.json"))


def save_arm(df, token, arms):
    arms.mkdir(parents=True, exist_ok=True)
    tmp = C.output_path(arms / (token + ".tmp.parquet"))
    df.to_parquet(tmp, index=False)
    tmp.replace(arms / (token + ".parquet"))
    c = C.condition_table(df)
    C.save_csv(c, arms / (token + ".conditions.csv"))
    if len(c) % 15:
        raise ValueError("expected 15 conditions per method")
    return c


def prepare(S, p, out):
    S_, m, cfg, fit, cal, dev, test = C.load_reference(p)
    from methods import fixed as MF
    d0, gen = MF.dev_pair_scores(dev)
    grid = S.SIM.grid_values()
    fixed = {
        "DBpHash fixed": S.SIM.grid_ceil(0.63075),
        "Youden J": MF.fit_youden(d0, gen, grid),
        "Equal error rate": MF.fit_eer(d0, gen, grid),
        "Otsu": MF.fit_otsu(d0, gen, grid),
    }
    C.atomic_json(out / "fixed_thresholds.json", {"thresholds": fixed})
    for b in test:
        for d in b.reg.values():
            d["_sorted_normal"] = np.sort(d["Z_eval"])
            names = sorted(set(S.CR.variant_names()) | set(map(str, np.unique(d["pos_variant"]))))
            d["_sorted_positive"] = {x: np.sort(d["Z_pos"][d["pos_variant"] == x]) for x in names}
            base = S.EV.score(fixed["DBpHash fixed"], d, 0.05)
            d["_baseline_BA23"] = (np.mean([base["TPR_" + x] for x in S.CR.variant_names()]) + 1 - base["FAR"]) / 2
    for b in test[:20]:
        for d in b.reg.values():
            for tau in (-1.0, fixed["DBpHash fixed"], fixed["Youden J"], 0.9, np.inf):
                slow = S.EV.score(tau, d, 0.05)
                fast = fast_eval(tau, d, 0.05)
                for f in ["FAR"] + ["TPR_" + x for x in S.CR.variant_names()]:
                    if slow[f] != fast[f]:
                        raise AssertionError((f, slow[f], fast[f]))
    return (fit, cal, dev, test, fixed)


def main(args, out):
    S = C.runner()
    S.encode = encode(S)
    ns = (100,) if args.smoke else NS
    betas = (0.95,) if args.smoke else BETAS
    modes = ("base_set",) if args.smoke else MODES
    seeds = C.SEEDS[:1] if args.smoke else C.SEEDS
    arms = C.output_path(out / "arms") if args.smoke else C.ARMS
    sel_jobs = [(b, v) for b in ((0.95,) if args.smoke else BETAS) for v in (("normalized15",) if args.smoke else ("raw15", "normalized15"))]
    mroot = C.OUT_ROOT / "models_smoke" if args.smoke else C.ROOT / "models"

    fit, cal, dev, test, fixed = prepare(S, None, out)

    sel = []
    STATE["N"] = 100
    S.CACHE = C.output_path(mroot / "selection_check")
    for beta, v in sel_jobs:
        a = round(1 - beta, 2)
        _, md, _ = S.model("base_set", v, a, 42, fit, dev)
        sel.append(dict(alpha=a, variant=v, development_mse=md["development_mse"]))

    sd = pd.DataFrame(sel)
    C.save_csv(sd, out / "selection_check.csv")
    choice = {str(a): g.sort_values("development_mse").variant.iloc[0] for a, g in sd.groupby("alpha")} if len(sd) else {}
    for n in ns:
        STATE["N"] = n
        S.CACHE = C.output_path(mroot / f"n{n}")
        for beta in betas:
            a = round(1 - beta, 2)
            tag = f"b{int(round(beta * 100)):02d}"
            token = f"baseline_n{n}_{tag}"
            if not arm_complete(arms, token):
                bf = S.FC.beta_fraction(a)
                rank = (bf.numerator * (n + 1) + bf.denominator - 1) // bf.denominator
                valid = [j for j in range(1, n + 1) if binom.sf(j - 1, n, beta) <= 0.05]
                k = min(valid) if valid else n + 1
                rows = []
                for b in test:
                    for kind, d in b.reg.items():
                        z = np.sort(d["Z_sup"][:n])
                        taus = dict(fixed)
                        taus["Direct split CP"] = S.SIM.strict_successor(float(z[rank - 1])) if rank <= n else np.inf
                        taus["Direct NP"] = S.SIM.strict_successor(float(z[k - 1])) if k <= n else np.inf
                        for name, tau in taus.items():
                            rows.append(score(S, b, kind, d, tau, n, a, name))
                save_arm(pd.DataFrame(rows), token, arms)
                C.atomic_json(
                    arms / (token + ".metadata.json"),
                    dict(n=n, beta=beta, boundary_case=False, cp_rank=rank, np_rank=k),
                )

            else:

                pass
            for mode in modes:
                for s in seeds:
                    tok = f"n{n}_{tag}_{mode}_s{s}"
                    if all(arm_complete(arms, f"{tok}_{c}") for c in CAL):


                        continue
                    t0 = time.perf_counter()
                    obj, md, path = S.model(mode, C.VAR[mode], a, s, fit, dev)
                    if md["epochs"] != 15 or md["pool"] != "mean" or md.get("reused"):
                        raise ValueError("stored model settings differ (epochs, pool or reused)")

                    tt = time.perf_counter() - t0
                    for c in CAL:
                        arm = f"{tok}_{c}"
                        if arm_complete(arms, arm):

                            continue
                        corr = S.calibration(obj, mode, a, cal, c)
                        lam = corr["global"]["Lambda"]
                        rows = []
                        for b in test:
                            for kind, d in b.reg.items():
                                q = S.predict(obj, d, a, mode, b.domain)
                                tau = S.SIM.strict_successor(S.FC.corrected_boundary(q, lam))
                                rows.append(score(S, b, kind, d, tau, n, a, f"{C.METHODS[mode]} + {c}", s, q, lam))
                        df = pd.DataFrame(rows)
                        if len(df) != 1016:
                            raise ValueError("expected 1016 registries")
                        save_arm(df, arm, arms)
                        C.atomic_json(
                            arms / f"{arm}.metadata.json",
                            dict(
                                n=n,
                                beta=beta,
                                alpha=a,
                                mode=mode,
                                variant=C.VAR[mode],
                                training_seed=s,
                                model_path=str(path),
                                model_sha256=C.sha256(path / "model.pt"),
                                metadata=md,
                                calibration=corr,
                                timing={"training_s": tt},
                            ),
                        )

    expected = []
    for n in ns:
        for beta in betas:
            tag = f"b{int(round(beta * 100)):02d}"
            expected.append(f"baseline_n{n}_{tag}")
            expected.extend(f"n{n}_{tag}_{mode}_s{seed}_{calibration}"
                            for mode in modes for seed in seeds for calibration in CAL)
    missing = [token for token in expected if not arm_complete(arms, token)]
    if missing:
        raise RuntimeError(f"Missing numerical results: {missing}")
    files = [arms / (token + ".conditions.csv") for token in sorted(expected)]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    C.save_csv(d, out / "conditions.csv")
    summ = C.method_summary(d.assign(training_seed=d.training_seed.fillna(-1)))
    C.save_csv(summ, out / "summary.csv")

    return {"conditions": int(len(d)), "fixed_thresholds": fixed, "selection_lower_dev_mse": choice}
if __name__ == "__main__":
    C.run_step("main", "Train, calibrate and evaluate", main)
