# Repeat the main experiment on pHash scores
from __future__ import annotations
import json, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binom
import experiment as C
NS = (100, 10)
ALPHAS = (0.01, 0.05)
MODE = "base_set"
CALS = ("fisch_global", "np_global")


def cache_dir(scheme, smoke):
    return C.output_path(C.OUT_ROOT / "data" / f"{scheme}_tasks_smoke" if smoke else C.ROOT / "data" / f"{scheme}_tasks")


def task_token(key):
    return hashlib.sha256(json.dumps(list(key)).encode()).hexdigest()


def build(args, out):
    S, MF, T, REG = C.modules()
    scheme = args.scheme
    if scheme != "phash":
        raise ValueError("scheme must be phash")

    _, m, cfg = S.X.reference_inputs(S.X.DEFAULT)

    dbp, _ = S.X.load_tasks(m)
    dbp = [b for b in dbp if b.scheme == "dbp"]
    ref = {(b.role, b.domain, b.structure, int(b.M), int(b.seed)): b for b in dbp}
    keys = sorted(ref)
    if args.smoke:
        keys = [k for r in ("train", "dev", "test") for k in [x for x in keys if x[0] == r][:3]]

    root = Path(cfg["root"])
    table = T.load_role_table(m["role_table"])

    pools, variants = ({}, {})
    for name, pc in cfg["pools"].items():
        pools[name] = REG.Pool(root / pc["cache"], name=name, scheme=scheme)
        frames = []
        for key in ("variants", "variants2"):
            rel = pc.get(key)
            if rel is None and key == "variants2" and pc.get("variants"):
                rel = str(pc["variants"]).replace("_var.parquet", "_var2.parquet")
            if rel and (root / rel).exists():
                v = pd.read_parquet(root / rel)
                frames.append(v[v.ok & (v.variant != "orig")])
        if frames:
            variants[name] = pd.concat(frames, ignore_index=True)
    dom_pool, dom_var, dom_roles = ({}, {}, {})
    for dom, d in cfg["domains"].items():
        names = d["pools"]
        dom_pool[dom] = pools[names[0]] if len(names) == 1 else REG.Pool([root / cfg["pools"][n]["cache"] for n in names], name=dom, scheme=scheme)
        dom_var[dom] = pd.concat([variants[n] for n in names if n in variants], ignore_index=True) if any(n in variants for n in names) else None
        dom_roles[dom] = T.roles_from_table(dom_pool[dom], table)
    for name, pc in cfg["pools"].items():
        n_dbp = REG.Pool(root / pc["cache"], name=name, scheme="dbp").n
        if n_dbp != pools[name].n:
            raise RuntimeError(f"pool {name}: {scheme} has {pools[name].n} images, dbp {n_dbp}")

    folder = cache_dir(scheme, args.smoke)
    folder.mkdir(parents=True, exist_ok=True)
    old_n = T.N_LIST[:]
    T.N_LIST = [100]
    rp_cache = {}
    mism = []
    try:
        for i, key in enumerate(keys):
            role, dom, struct, M, seed = key
            tok = task_token([scheme] + list(key))
            npz = folder / (tok + ".npz")
            rec = folder / (tok + ".complete.json")
            if rec.exists():

                continue
            if (dom, role) not in rp_cache:
                rp_cache[dom, role] = T.role_pool(dom_pool[dom], dom_roles[dom], role)
            b = T.build_bundle(rp_cache[dom, role], dom_var[dom], role, dom, struct, M, seed, cfg)
            if b.note:
                raise RuntimeError(f"task could not be built for {key}: {b.note}")
            r = ref[key]
            same = list(map(str, b.sup_files)) == list(map(str, r.sup_files)) and list(map(str, b.eval_files)) == list(map(str, r.eval_files)) and (set(b.reg) == set(r.reg))
            if not same:
                mism.append(list(key))
                raise RuntimeError(f"{scheme} task sampled different files than DBpHash for {key}; stopped early")
            tmp = folder / (tok + ".pending.npz")
            T.save_bundle(b, tmp)
            Path(str(tmp)).replace(npz)
            Path(str(tmp) + ".json").replace(Path(str(npz) + ".json"))
            C.atomic_json(
                rec,
                {
                    "scheme": scheme,
                    "key": list(key),
                    "npz_sha256": C.sha256(npz),
                    "json_sha256": C.sha256(str(npz) + ".json"),
                    "same_files_as_dbphash": same,
                    "time": datetime.datetime.now().isoformat(),
                },
            )

    finally:
        T.N_LIST = old_n
    if mism:
        raise RuntimeError(f"{len(mism)} tasks sampled different files than DBpHash")
    return {"scheme": scheme, "tasks": len(keys), "cache": str(folder)}


def load_cache(T, folder):
    out = []
    for rec_p in sorted(folder.glob("*.complete.json")):
        rec = json.loads(rec_p.read_text(encoding="utf-8"))
        npz = rec_p.with_name(rec_p.name.replace(".complete.json", ".npz"))
        if C.sha256(npz) != rec["npz_sha256"] or C.sha256(str(npz) + ".json") != rec["json_sha256"]:
            raise RuntimeError(f"checksum mismatch {npz}")
        b = T.load_bundle(npz)
        meta = json.loads(Path(str(npz) + ".json").read_text(encoding="utf-8"))
        for f in T._SCALARS:
            if f in meta:
                setattr(b, f, meta[f])
        out.append(b)
    return out


def run(args, out):
    S, MF, T, REG = C.modules()
    scheme = args.scheme
    w = S.SIM.SCHEME_W[scheme]
    g = S.SIM.grid_values(w)
    ns = (100,) if args.smoke else NS
    alphas = (0.05,) if args.smoke else ALPHAS
    seeds = C.SEEDS[:1] if args.smoke else C.SEEDS

    bundles = load_cache(T, cache_dir(scheme, args.smoke))
    train = [b for b in bundles if b.role == "train"]
    fit, cal = S.R.split_train_cal(train, 0.5)
    _, m, cfg = S.X.reference_inputs(S.X.DEFAULT)
    dev = [b for b in bundles if b.role == "dev" and b.M in cfg["train_M"]]
    test = [b for b in bundles if b.role == "test"]
    if not args.smoke:
        assert (len(fit), len(cal), len(dev)) == (324, 324, 568), (len(fit), len(cal), len(dev))
        assert sum((len(b.reg) for b in test)) == 1016
    names = S.CR.variant_names()
    tau_h = MF.fit_C0h(w)
    C.prepare_test(test, tau_h, names)
    regs = [(b, k, d) for b in test for k, d in b.reg.items()]

    d0, gen = MF.dev_pair_scores(dev)
    fixed = {
        "Hash convention (10 bits)": tau_h,
        "Youden J": MF.fit_C0j(d0, gen, g),
        "Equal error rate": MF.fit_C0eer(d0, gen, g),
        "Otsu": MF.fit_C0otsu(d0, gen, g),
    }
    C.atomic_json(out / "fixed_thresholds.json", fixed)
    rows = []
    for n in ns:
        for a in alphas:
            beta = 1 - a
            frac = S.FC.beta_fraction(a)
            rank = (frac.numerator * (n + 1) + frac.denominator - 1) // frac.denominator
            valid = [j for j in range(1, n + 1) if binom.sf(j - 1, n, beta) <= 0.05]
            k_np = min(valid) if valid else n + 1
            for b, k, d in regs:
                z = np.sort(np.asarray(d["Z_sup"], float)[:n])
                taus = dict(fixed)
                taus["Direct split CP"] = C.succ(float(z[rank - 1]), g) if rank <= n else float("inf")
                taus["Direct NP"] = C.succ(float(z[k_np - 1]), g) if k_np <= n else float("inf")
                for name, tau in taus.items():
                    rows.append(C.row(b, k, d, tau, n, a, name, names, scheme=scheme))

    cal_info = []
    for n in ns:
        S.CACHE = C.output_path((C.OUT_ROOT / f"models_{scheme}_smoke" if args.smoke else C.ROOT / f"models_{scheme}") / f"n{n}")
        S.encode = (lambda nn: lambda d, a, mode, domain: C.encode(S, d, a, mode, nn))(n)
        for a in alphas:
            for s in seeds:
                obj, md, path = S.model(MODE, C.VAR[MODE], a, s, fit, dev, None)

                for c in CALS:
                    if c.startswith("fisch") and len(cal) < S.FC.calibration_min_tasks(a):
                        if not args.smoke:
                            raise RuntimeError(f"too few calibration tasks for Fisch calibration: {len(cal)}")
                        cal_info.append(dict(n=n, alpha=a, training_seed=s, calibration=c, skipped="smoke: too few calibration tasks"))

                        continue
                    info = S.calibration(obj, MODE, a, cal, c)
                    lam = info["global"]["Lambda"]
                    cal_info.append(dict(n=n, alpha=a, training_seed=s, calibration=c, Lambda=lam, model=str(path)))
                    for b, k, d in regs:
                        q = S.predict(obj, d, a, MODE, b.domain)
                        tau = C.succ(S.FC.corrected_boundary(q, lam), g)
                        rows.append(C.row(b, k, d, tau, n, a, f"Meta set + {c}", names, seed=s, scheme=scheme, q_pred=q, Lambda=lam))

    R = pd.DataFrame(rows)
    C.save_parquet(R, out / "rows.parquet")
    C.atomic_json(out / "calibration.json", cal_info)
    cond = C.condition_table(R)
    C.save_csv(cond, out / "conditions.csv")
    summ = C.method_summary(cond)
    C.save_csv(summ, out / "summary.csv")
    judg = []
    for (n, a), gsub in summ[summ.method == "Meta set + np_global"].groupby(["n", "alpha"]):
        V = float(gsub.V.iloc[0])
        judg.append(dict(n=n, alpha=a, V=V, judgment="PASS" if V <= C.DELTA + 1e-12 else "NON PASS"))

    return {"scheme": scheme, "judgment": judg}


def main(args, out):
    if args.part == "build":
        return build(args, out)
    if args.part == "run":
        return run(args, out)
    raise ValueError("--part build | run")
if __name__ == "__main__":
    import sys
    part = "build" if "--part" in sys.argv and sys.argv[sys.argv.index("--part") + 1] == "build" else "run"
    C.run_step("phash_build" if part == "build" else "phash", "pHash experiments", main)
