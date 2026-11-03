# External test on the DISC21 ground truth
from __future__ import annotations
import json, zipfile, datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binom
import experiment as C
from experiments.support_baselines import np_rank
from experiments.prepare_disc21 import FeatStore, VAL_LIST, QZIP, GT
NS = (100, 10)
ALPHAS = (0.01, 0.05)
M_BASE = (10, 1000)
GROW = 10
REG_SEEDS = tuple(range(60))
GT_CAP = 200
N_SUP = 100
SEED0 = 20261017
DBP_MODES = {"base_set": "Meta set", "base_summary256": "Summary H256"}
CALS = ("np_global", "fisch_global")
FIXED_DBP = ("DBpHash fixed", "Youden J", "Equal error rate", "Otsu")
SD_FLOOR_SSCD = 1e-06
DOMAIN = "disc21_official"


def features(smoke):
    refs = [Path(s.strip()).stem for s in VAL_LIST.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not smoke:
        rs = FeatStore.open_read(C.ROOT / "data" / "disc21_features" / "references")
        qs = FeatStore.open_read(C.ROOT / "data" / "disc21_features" / "queries")
        if list(rs.index.name) != refs:
            raise RuntimeError("reference list differs from disc21_prepare")
        return dict(
            r_name=np.array(refs),
            r_ok=rs.ok.copy(),
            r_mid=np.asarray(rs.mid),
            r_high=np.asarray(rs.high),
            r_emb=np.asarray(rs.emb, np.float32),
            q_name=np.array(list(qs.index.name)),
            q_ok=qs.ok.copy(),
            q_mid=np.asarray(qs.mid),
            q_high=np.asarray(qs.high),
            q_emb=np.asarray(qs.emb, np.float32),
            synthetic=False,
        )
    with zipfile.ZipFile(QZIP) as z:
        qn = sorted((Path(n).stem for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))))
    rng = np.random.default_rng(0)

    def unit(n):
        x = rng.normal(size=(n, 512)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    u64 = lambda n: rng.integers(0, np.iinfo(np.uint64).max, n, dtype=np.uint64, endpoint=True)
    return dict(
        r_name=np.array(refs),
        r_ok=np.ones(len(refs), bool),
        r_mid=u64(len(refs)),
        r_high=u64(len(refs)),
        r_emb=unit(len(refs)),
        q_name=np.array(qn),
        q_ok=np.ones(len(qn), bool),
        q_mid=u64(len(qn)),
        q_high=u64(len(qn)),
        q_emb=unit(len(qn)),
        synthetic=True,
    )


def design(F):
    gt = pd.read_csv(GT, header=None, names=["query", "reference"])
    rpos = {n: i for i, n in enumerate(F["r_name"])}
    qpos = {n: i for i, n in enumerate(F["q_name"])}
    gtq = set(gt["query"])
    g = gt[gt.reference.isin(rpos) & gt["query"].isin(qpos)].copy()
    g["ri"] = g.reference.map(rpos)
    g["qi"] = g["query"].map(qpos)
    g = g[F["r_ok"][g.ri.to_numpy()] & F["q_ok"][g.qi.to_numpy()]]
    gt_refs = np.array(sorted(set(g.ri)))
    pos_of = g.groupby("ri").qi.apply(lambda s: np.sort(s.to_numpy())).to_dict()
    nongt = np.array(sorted(set(np.flatnonzero(F["r_ok"])) - set(gt.reference.map(rpos).dropna().astype(int))))
    normals = np.array(sorted((i for i, n in enumerate(F["q_name"]) if n not in gtq and F["q_ok"][i])))
    return dict(
        gt_refs=gt_refs,
        pos_of=pos_of,
        nongt=nongt,
        normals=normals,
        counts={
            "gt_pairs_in_heldout_half": int(len(g)),
            "gt_references": int(len(gt_refs)),
            "non_gt_references": int(len(nongt)),
            "normal_queries": int(len(normals)),
        },
    )


def registry(Dz, M, seed):
    rng = np.random.default_rng([SEED0, M, seed])
    k = min(GT_CAP, M // 2)
    gsel = np.sort(rng.choice(Dz["gt_refs"], k, replace=False))
    fill = rng.choice(Dz["nongt"], M - k + (GROW - 1) * M, replace=False)
    base = np.concatenate([gsel, fill[:M - k]])
    grown = np.concatenate([base, fill[M - k:]])
    perm = rng.permutation(len(Dz["normals"]))
    sup = Dz["normals"][perm[:N_SUP]]
    ev = Dz["normals"][perm[N_SUP:]]
    pos = np.concatenate([Dz["pos_of"][r] for r in gsel])
    return ({"base": base, "grow_i": grown}, sup, ev, pos)


def load_model_dir(S, path: Path, mode, alpha, seed):
    import torch
    md = json.loads((path / "metadata.json").read_text(encoding="utf-8-sig"))
    if md["mode"] != mode or md["alpha"] != alpha or md["training_seed"] != seed:
        raise ValueError(f"stored model metadata does not match mode={mode}, alpha={alpha}, seed={seed}: {path}")
    torch.manual_seed(seed)
    obj = S.Predictor(mode, md["mu"], md["sd"], md["pool"])
    obj.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    obj.eval()
    return (obj, C.sha256(path / "model.pt"))


# All thresholds / models / Lambdas, read from earlier steps
def load_models(S, smoke, ns, alphas, seeds):
    fz = {"dbp": {"fixed": {}, "zc": {}, "models": {}}, "sscd": {"fixed": {}, "zc": {}, "models": {}}, "sources": {}}
    B = pd.read_parquet(C.arm_path(100, 0.95, baseline=True))
    for m in FIXED_DBP:
        t = B.loc[B.method == m, "tau"].unique()
        if len(t) != 1:
            raise ValueError(f"fixed rule {m} is not a single threshold")
        fz["dbp"]["fixed"][m] = float(t[0])
    hr = json.loads(C.stored(C.RESULTS / "hash_thresholds" / "hash_rule.json").read_text(encoding="utf-8"))
    fz["dbp"]["fixed"]["Hash convention"] = float(hr["tau_grid"])
    fz["sources"]["dbp_fixed"] = [str(C.arm_path(100, 0.95, baseline=True)), str(C.stored(C.RESULTS / "hash_thresholds" / "hash_rule.json"))]
    a2 = json.loads(C.stored(C.RESULTS / "support_baselines" / "fitted_values.json").read_text(encoding="utf-8"))
    for e in a2:
        if e["method"] == "Z-norm + NP rank":
            fz["dbp"]["zc"][e["n"], e["alpha"]] = float(e["c"])
    fz["sources"]["dbp_znorm"] = str(C.stored(C.RESULTS / "support_baselines" / "fitted_values.json"))
    for n in ns:
        for a in alphas:
            for mode in DBP_MODES:
                for s in seeds:
                    obj, md, path = C.load_model(S, n, mode, a, s)
                    lam = {}
                    for c in CALS:
                        mp = Path(str(C.arm_path(n, round(1 - a, 2), mode, s, c)).replace(".parquet", ".metadata.json"))
                        meta = json.loads(mp.read_text(encoding="utf-8-sig"))
                        lam[c] = float(meta["calibration"]["global"]["Lambda"])
                    fz["dbp"]["models"][n, a, mode, s] = (obj, lam)
    sr = C.RESULTS / ("smoke" if smoke else "") / "sscd"
    if not all((sr / f).is_file() for f in ("fixed_thresholds.json", "calibration.json", "summary.csv")):
        raise FileNotFoundError(f"run the sscd step first ({sr})")
    fz["sscd"]["fixed"] = json.loads((sr / "fixed_thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    cal = json.loads((sr / "calibration.json").read_text(encoding="utf-8"))
    lam = {}
    for e in cal:
        if e["method"] == "Z-norm + NP rank":
            fz["sscd"]["zc"][e["n"], e["alpha"]] = float(e["c"])
        elif e["method"].startswith("Meta set + "):
            lam[e["n"], e["alpha"], e["training_seed"], e["method"][len("Meta set + "):]] = float(e["Lambda"])
    mroot = C.OUT_ROOT / "models_sscd_smoke" if smoke else C.ROOT / "models_sscd"
    for n in ns:
        for a in alphas:
            for s in seeds:
                obj, _ = load_model_dir(
                    S,
                    mroot / f"n{n}" / f"base_set__{C.VAR['base_set']}__a{a}__s{s}",
                    "base_set",
                    a,
                    s,
                )
                fz["sscd"]["models"][n, a, "base_set", s] = (obj, {c: lam[n, a, s, c] for c in CALS if (n, a, s, c) in lam})
    fz["sources"]["sscd"] = [str(sr / "fixed_thresholds.json"), str(sr / "calibration.json"), str(mroot)]
    return fz


# Every method's threshold for one registry (decision S >= tau)
def taus(S, fz, scheme, z_sup, M_eff, n, a, g):
    up = (lambda u: C.succ(u, g)) if scheme == "dbp" else C.cont_succ
    floor = float(np.min(np.diff(g))) if scheme == "dbp" else SD_FLOOR_SSCD
    out = [(m, t, -1) for m, t in fz[scheme]["fixed"].items()]
    z = np.sort(np.asarray(z_sup, float)[:n])
    frac = S.FC.beta_fraction(a)
    r_cp = (frac.numerator * (n + 1) + frac.denominator - 1) // frac.denominator
    valid = [j for j in range(1, n + 1) if binom.sf(j - 1, n, 1 - a) <= 0.05]
    r_np = min(valid) if valid else n + 1
    out.append(("Direct split CP", up(float(z[r_cp - 1])) if r_cp <= n else float("inf"), -1))
    out.append(("Direct NP", up(float(z[r_np - 1])) if r_np <= n else float("inf"), -1))
    c = fz[scheme]["zc"][n, a]
    out.append(("Z-norm + NP rank", up(z.mean() + c * max(z.std(), floor)) if np.isfinite(c) else float("inf"), -1))
    d = {"Z_sup": np.asarray(z_sup, float), "M": M_eff}
    for (nn, aa, mode, s), (obj, lam) in fz[scheme]["models"].items():
        if (nn, aa) != (n, a):
            continue
        q = C.predict(S, obj, d, a, mode, n)
        name = DBP_MODES[mode] if scheme == "dbp" else "Meta set"
        for cal, L in lam.items():
            out.append((f"{name} + {cal}", up(S.FC.corrected_boundary(q, L)), s))
    return out


def score_rows(meta, tau_list, z_ev, z_pos, a, base_name):
    zz = np.sort(z_ev)
    zp = np.sort(z_pos)
    rows = []
    for m, t, s in tau_list:
        sent = not np.isfinite(t)
        far = 0.0 if sent else (len(zz) - np.searchsorted(zz, t, side="left")) / len(zz)
        tpr = 0.0 if sent else (len(zp) - np.searchsorted(zp, t, side="left")) / len(zp)
        rows.append(dict(
            meta,
            method=m,
            training_seed=s,
            tau=float(t),
            sentinel=sent,
            FAR=far,
            violation=far > a + 1e-12,
            excess=far - a,
            n_eval=len(zz),
            n_pos=len(zp),
            TPR23=tpr,
            BA23=(tpr + 1 - far) / 2,
        ))
    b = [r["BA23"] for r in rows if r["method"] == base_name]
    if len(b) != 1:
        raise ValueError("baseline rule missing")
    for r in rows:
        r["baseline_BA23"] = b[0]
    return rows
PROTOCOL = {
    "delta": C.DELTA,
    "M_base": list(M_BASE),
    "growth_factor": GROW,
    "registry_seeds": len(REG_SEEDS),
    "gt_cap": GT_CAP,
    "gt_registry_divisor": 2,
    "support": N_SUP,
    "n": list(NS),
    "alphas": list(ALPHAS),
    "training_seeds": list(C.SEEDS),
    "seed0": SEED0,
    "risk_metric": "V",
    "comparison_metric": "passing",
    "split_reused": True,
    "reference_methods": {"dbp": "DBpHash fixed", "sscd": "Equal error rate"},
}


def main(args, out):
    S, MF, T, REG = C.modules()
    smoke = args.smoke
    ns = (100,) if smoke else NS
    alphas = (0.05,) if smoke else ALPHAS
    seeds = C.SEEDS[:1] if smoke else C.SEEDS
    regseeds = REG_SEEDS[:3] if smoke else REG_SEEDS
    if not smoke and (out / "summary.csv").exists():
        raise RuntimeError("disc21 already completed; the result folder is kept")

    fz = load_models(S, smoke, ns, alphas, seeds)
    C.atomic_json(
        out / "model_inputs.json",
        {
            "dbp_fixed": fz["dbp"]["fixed"],
            "sscd_fixed": fz["sscd"]["fixed"],
            "dbp_znorm_c": {f"n{k[0]}_a{k[1]}": v for k, v in fz["dbp"]["zc"].items()},
            "sscd_znorm_c": {f"n{k[0]}_a{k[1]}": v for k, v in fz["sscd"]["zc"].items()},
            "lambdas": {f"{sc}_n{k[0]}_a{k[1]}_{k[2]}_s{k[3]}": v[1] for sc in ("dbp", "sscd") for k, v in fz[sc]["models"].items()},
            "sources": fz["sources"],
        },
    )
    F = features(smoke)
    Dz = design(F)
    C.atomic_json(out / "design_counts.json", {**Dz["counts"], "synthetic_features": F["synthetic"]})
    g = C.grid()

    parts = C.output_path(out / "parts")
    parts.mkdir(parents=True, exist_ok=True)
    scores = C.output_path(out / "scores")
    scores.mkdir(parents=True, exist_ok=True)
    for M in M_BASE:
        for rs in regseeds:
            pp = parts / f"M{M}_s{rs}.parquet"
            if pp.exists():

                continue
            regs, sup, ev, pos = registry(Dz, M, rs)
            qidx = np.concatenate([sup, ev, pos])
            ns_, ne_ = (len(sup), len(ev))
            rows, keep = ([], {})
            for kind, R in regs.items():
                Zd = S.SIM.query_max(F["q_mid"][qidx], F["q_high"][qidx], F["r_mid"][R], F["r_high"][R])["Z"]
                Zs = C.max_cos(F["q_emb"][qidx], F["r_emb"][R])
                for scheme, Z, base_name in (("dbp", Zd, "DBpHash fixed"), ("sscd", Zs, "Equal error rate")):
                    z_sup, z_ev, z_pos = (Z[:ns_], Z[ns_:ns_ + ne_], Z[ns_ + ne_:])
                    keep[f"{scheme}_{kind}"] = Z
                    for n in ns:
                        for a in alphas:
                            meta = dict(
                                domain=DOMAIN,
                                structure="random",
                                kind=kind,
                                M=M,
                                M_eff=len(R),
                                scheme=scheme,
                                n=n,
                                alpha=a,
                                beta=round(1 - a, 2),
                                seed=rs,
                                boundary_case=False,
                            )
                            rows += score_rows(
                                meta,
                                taus(S, fz, scheme, z_sup, len(R), n, a, g),
                                z_ev,
                                z_pos,
                                a,
                                base_name,
                            )
            np.savez_compressed(
                C.output_path(scores / f"M{M}_s{rs}.npz"),
                base=regs["base"].astype(np.int32),
                grow_i=regs["grow_i"].astype(np.int32),
                support=sup.astype(np.int32),
                evaluation=ev.astype(np.int32),
                positives=pos.astype(np.int32),
                **{k: v.astype(np.float64) for k, v in keep.items()},
            )
            C.save_parquet(pd.DataFrame(rows), pp)

    R = pd.concat(
        [pd.read_parquet(parts / f"M{M}_s{rs}.parquet") for M in M_BASE for rs in regseeds],
        ignore_index=True,
    )
    C.save_parquet(R, out / "rows.parquet")
    cond = C.condition_table(R)
    C.save_csv(cond, out / "conditions.csv")
    summ = C.method_summary(cond, by=("scheme", "method", "n", "alpha"))
    C.save_csv(summ, out / "summary.csv")
    j = []
    for scheme, tag, ref in (("dbp", "", "DBpHash fixed"), ("sscd", "s", "Equal error rate")):
        sm = summ[summ.scheme == scheme].set_index(["method", "n", "alpha"])
        for n in ns:
            for a in alphas:
                v = float(sm.loc[("Meta set + np_global", n, a), "V"])
                j.append(dict(
                    hypothesis=f"DISC21-H2{tag}",
                    scheme=scheme,
                    n=n,
                    alpha=a,
                    V=v,
                    judgment="PASS" if v <= C.DELTA + 1e-12 else "NON PASS",
                ))
        for a in alphas:
            me = float(sm.loc[("Meta set + np_global", ns[0], a), "passing"])
            rf = float(sm.loc[(ref, ns[0], a), "passing"])
            j.append(dict(
                hypothesis=f"DISC21-H1{tag}",
                scheme=scheme,
                n=ns[0],
                alpha=a,
                proposed_passing=me,
                reference_rule=ref,
                reference_passing=rf,
                conditions=int(sm.loc[(ref, ns[0], a), "conditions"]),
                judgment="PASS" if me > rf else "NON PASS",
            ))
    ok = lambda sc: "PASS" if all(x["judgment"] == "PASS" for x in j if x["scheme"] == sc) else "NON PASS"
    res = {
        "results": j,
        "dbp_pass": ok("dbp"),
        "sscd_pass": ok("sscd"),
        "smoke": smoke,
        "synthetic_features": F["synthetic"],
    }

    return {"dbp_pass": res["dbp_pass"], "sscd_pass": res["sscd_pass"], "smoke": smoke}
if __name__ == "__main__":
    C.run_step("disc21", "DISC21 공식 ground truth 외부 시험", main)
