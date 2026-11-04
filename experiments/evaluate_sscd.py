# Train, calibrate and evaluate on SSCD scores
from __future__ import annotations
import json, datetime
import numpy as np, pandas as pd
from scipy.stats import binom
import experiment as C
from experiments.phash import load_cache
from experiments.support_baselines import np_rank
NS = (100, 10)
ALPHAS = (0.01, 0.05)
MODE = "base_set"
CALS = ("fisch_global", "np_global")
SD_FLOOR = 1e-06


# Candidate thresholds and FPR / TPR for decision S >= t; negatives weighted equally across domains
def curves(neg_by_dom, pos):
    cand = np.unique(np.concatenate([np.concatenate(list(neg_by_dom.values())), pos]))
    fpr = np.zeros(len(cand))
    for z in neg_by_dom.values():
        z = np.sort(z)
        fpr += 1.0 - np.searchsorted(z, cand, side="left") / len(z)
    fpr /= len(neg_by_dom)
    ps = np.sort(pos)
    tpr = 1.0 - np.searchsorted(ps, cand, side="left") / len(ps)
    return (cand, fpr, tpr)


# Positives: S_gen of the photometric PRIMARY edits, the same set as methods.fixed (PHOTO) for DBpHash
def fixed_rules(dev, photo, smoke=False):
    neg = {}
    for b in dev:
        h = b.reg["base"]["hist_sup"]
        if 100 in h and len(h[100]):
            neg.setdefault(b.domain, []).append(np.asarray(h[100], float))
    neg = {k: np.concatenate(v) for k, v in neg.items()}
    pos = np.concatenate([np.zeros(0)] + [b.S_gen[np.isin(b.gen_variant, photo)] for b in dev if len(b.S_gen)])
    positive_set = list(photo)
    if not len(pos):
        if not smoke:
            raise RuntimeError("no photometric development positives for the SSCD fixed rules")
        pos = np.concatenate([b.S_gen for b in dev if len(b.S_gen)])
        positive_set = "smoke only: all stored edits"
    if not neg or not len(pos):
        raise RuntimeError("no development pairs for the SSCD fixed rules")
    cand, fpr, tpr = curves(neg, pos)
    rules = {
        "Youden J": float(cand[int(np.argmax(tpr - fpr))]),
        "Equal error rate": float(cand[int(np.argmin(np.abs(fpr - (1.0 - tpr))))]),
    }
    w_neg = np.concatenate([np.full(len(v), 1.0 / (len(v) * len(neg))) for v in neg.values()])
    vals = np.concatenate([np.concatenate(list(neg.values())), pos])
    wts = np.concatenate([0.5 * w_neg, np.full(len(pos), 0.5 / len(pos))])
    o = np.argsort(vals, kind="stable")
    v, p_ = (vals[o], wts[o])
    cw = np.cumsum(p_)
    cm = np.cumsum(p_ * v)
    mu = cm[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu * cw - cm) ** 2 / (cw * (1 - cw))
    sb[~np.isfinite(sb)] = -1.0
    k = int(np.argmax(sb))
    rules["Otsu"] = float(v[min(k + 1, len(v) - 1)])
    return (
        rules,
        {
            "negatives_per_domain": {k: int(len(x)) for k, x in neg.items()},
            "positives": int(len(pos)),
            "positive_set": positive_set,
        },
    )


def main(args, out):
    S, MF, T, REG = C.modules()
    ns = (100,) if args.smoke else NS
    alphas = (0.05,) if args.smoke else ALPHAS
    seeds = C.SEEDS[:1] if args.smoke else C.SEEDS

    bundles = load_cache(
        T,
        C.output_path(C.OUT_ROOT / "data" / "sscd_tasks_smoke" if args.smoke else C.ROOT / "data" / "sscd_tasks"),
    )
    train = [b for b in bundles if b.role == "train"]
    fit, cal = S.R.split_train_cal(train, 0.5)
    _, m, cfg = S.X.reference_inputs(S.X.DEFAULT)
    dev = [b for b in bundles if b.role == "dev" and b.M in cfg["train_M"]]
    test = [b for b in bundles if b.role == "test"]
    if not args.smoke:
        assert (len(fit), len(cal), len(dev)) == (324, 324, 568), (len(fit), len(cal), len(dev))
        assert sum((len(b.reg) for b in test)) == 1016
    names = S.CR.variant_names()
    fixed, fixinfo = fixed_rules(dev, MF.PHOTO, args.smoke)
    C.atomic_json(out / "fixed_thresholds.json", {"thresholds": fixed, **fixinfo})
    C.prepare_test(test, fixed["Equal error rate"], names)
    regs = [(b, k, d) for b in test for k, d in b.reg.items()]

    rows, info = ([], [])
    for n in ns:
        for a in alphas:
            beta = 1 - a
            frac = S.FC.beta_fraction(a)
            rank = (frac.numerator * (n + 1) + frac.denominator - 1) // frac.denominator
            valid = [j for j in range(1, n + 1) if binom.sf(j - 1, n, beta) <= 0.05]
            k_np = min(valid) if valid else n + 1
            t = []
            for b in cal:
                d = b.reg["base"]
                z = np.asarray(d["Z_sup"], float)[:n]
                t.append((C.target_quantile(S, d["Z_eval"], a) - z.mean()) / max(z.std(), SD_FLOOR))
            t = np.sort(t)
            kk = np_rank(len(t), C.DELTA)
            c = float(t[kk - 1]) if kk else float("inf")
            info.append(dict(method="Z-norm + NP rank", n=n, alpha=a, c=c, rank=kk))
            for b, k, d in regs:
                z = np.sort(np.asarray(d["Z_sup"], float)[:n])
                taus = dict(fixed)
                taus["Direct split CP"] = C.cont_succ(float(z[rank - 1])) if rank <= n else float("inf")
                taus["Direct NP"] = C.cont_succ(float(z[k_np - 1])) if k_np <= n else float("inf")
                taus["Z-norm + NP rank"] = C.cont_succ(z.mean() + c * max(z.std(), SD_FLOOR)) if np.isfinite(c) else float("inf")
                for name, tau in taus.items():
                    rows.append(C.row(b, k, d, tau, n, a, name, names, scheme="sscd"))

    for n in ns:
        S.CACHE = C.output_path((C.OUT_ROOT / "models_sscd_smoke" if args.smoke else C.ROOT / "models_sscd") / f"n{n}")
        S.encode = (lambda nn: lambda d, a, mode, domain: C.encode(S, d, a, mode, nn))(n)
        for a in alphas:
            for s in seeds:
                obj, md, path = S.model(MODE, C.VAR[MODE], a, s, fit, dev, None)

                for c in CALS:
                    if c.startswith("fisch") and len(cal) < S.FC.calibration_min_tasks(a):
                        if not args.smoke:
                            raise RuntimeError("too few calibration tasks")

                        continue
                    ci = S.calibration(obj, MODE, a, cal, c)
                    lam = ci["global"]["Lambda"]
                    info.append(dict(method=f"Meta set + {c}", n=n, alpha=a, training_seed=s, Lambda=lam, model=str(path)))
                    for b, k, d in regs:
                        q = S.predict(obj, d, a, MODE, b.domain)
                        tau = C.cont_succ(S.FC.corrected_boundary(q, lam))
                        rows.append(C.row(b, k, d, tau, n, a, f"Meta set + {c}", names, seed=s, scheme="sscd", q_pred=q, Lambda=lam))

    R = pd.DataFrame(rows)
    C.save_parquet(R, out / "rows.parquet")
    C.atomic_json(out / "calibration.json", info)
    cond = C.condition_table(R)
    C.save_csv(cond, out / "conditions.csv")
    summ = C.method_summary(cond)
    C.save_csv(summ, out / "summary.csv")
    j = []
    for (n, a), g in summ[summ.method == "Meta set + np_global"].groupby(["n", "alpha"]):
        j.append(dict(hypothesis="H2", n=n, alpha=a, V=float(g.V.iloc[0]), judgment="PASS" if g.V.iloc[0] <= C.DELTA + 1e-12 else "NON PASS"))
    for a in alphas:
        s100 = summ[(summ.n == ns[0]) & (summ.alpha == a)].set_index("method")
        if "Meta set + np_global" in s100.index:
            me = float(s100.loc["Meta set + np_global", "far_ok"])
            others = {m_: float(s100.loc[m_, "far_ok"]) for m_ in ("Youden J", "Equal error rate", "Otsu") if m_ in s100.index}
            j.append(dict(
                hypothesis="H1",
                n=ns[0],
                alpha=a,
                proposed_far_ok=me,
                fixed_far_ok=others,
                judgment="PASS" if all((me > v for v in others.values())) else "NON PASS",
            ))

    return {"judgment": j}
if __name__ == "__main__":
    C.run_step("sscd", "Train and evaluate SSCD thresholds", main)
