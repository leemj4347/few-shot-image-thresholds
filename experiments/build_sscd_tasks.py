# Build the SSCD task bundles matched to the DBpHash tasks
from __future__ import annotations
import json, datetime
from pathlib import Path
import numpy as np, pandas as pd
import experiment as C
from experiments.phash import task_token


# Build registry tasks with SSCD scores
def build_one(T, pool, variants, role, domain, structure, M, seed, cfg, lookup, E):
    b = T.Bundle(role=role, domain=domain, structure=structure, M=M, seed=seed, scheme="sscd")
    minr = cfg.get("holdout_min_reps", 50)
    need = cfg.get("query_holdout_need")
    hold = T.holdout_collections(
        pool,
        seed,
        cfg.get("query_collection_frac", 0.3),
        minr,
        need,
    ) if pool.has_coll else set()
    gf = cfg.get("growth", {}).get("factor", 10)
    nc = cfg.get("growth", {}).get("new_collections", 3)
    reserve = T.reserve_collections(pool, seed, nc, hold, minr)
    try:
        reg = T.Registry.sample(pool, M, seed, structure, exclude_coll=hold | reserve)
    except ValueError as e:
        b.note = f"registry: {e}"
        return b
    if variants is not None and len(variants):
        req = cfg.get("required_variants")
        if req:
            req = [v for v in req if v in set(variants.variant.unique())]
            cnt = variants[variants.variant.isin(req)].groupby("file").variant.nunique()
            full = set(cnt[cnt >= len(req)].index)
        else:
            full = set(variants.file.unique())
        want = min(int(cfg.get("min_variant_members", 20)), max(1, M // 2))
        if full:
            reg = reg.ensure_files(full, want, seed)
    grown = {}
    try:
        grown["grow_i"] = reg.grow(M * gf)
    except ValueError:
        pass
    if reserve:
        try:
            grown["grow_ii"] = reg.add_collections(nc, seed, per_coll=cfg.get("growth", {}).get("per_collection"), allowed=reserve)
        except ValueError:
            pass
    allidx = np.unique(np.concatenate([reg.idx] + [g.idx for g in grown.values()])) if grown else reg.idx
    tmp = T.Registry(pool, allidx, structure, seed, 0, exclude_coll=hold)
    want = pool.n if cfg.get("eval_all") else T.N_MAX + T.N_EVAL
    q = T.sample_queries(pool, tmp, want, seed, exclude_files=cfg.get("_exclude_files") or None)
    if len(q) < cfg.get("n_min_total", 120):
        b.note = f"queries {len(q)} < min"
        return b
    if cfg.get("support_random"):
        q = q[np.random.default_rng(seed + 90001).permutation(len(q))]
    n_sup = min(T.N_MAX, len(q) - 1) if cfg.get("eval_all") else min(T.N_MAX, len(q) // 3)
    sup_idx, eval_idx = (q[:n_sup], q[n_sup:])
    b.n_sup = len(sup_idx)
    b.n_eval = len(eval_idx)
    b.n_coll_registry = reg.n_collections()
    b.sup_idx = sup_idx
    b.eval_idx = eval_idx
    b.hold = set(hold)
    b.reserve = set(reserve)
    b.sup_files = pool.df.file.to_numpy()[sup_idx]
    b.eval_files = pool.df.file.to_numpy()[eval_idx]
    pv = T.positives_for(reg, variants, cfg.get("n_pos_originals", 200), seed) if variants is not None else None
    if pv is not None and len(pv):
        Ev = lookup.var(pv)
        Eo = lookup.orig_files(pv)
        b.S_gen = np.einsum("ij,ij->i", Ev.astype(np.float32), Eo.astype(np.float32)).astype(np.float64)
        b.gen_variant = pv.variant.to_numpy()
    cap = int(cfg.get("train_eval_cap", 0)) if role == "train" else 0
    ev = eval_idx
    if cap and len(eval_idx) > cap:
        pick = np.sort(np.random.default_rng(seed + 7777).choice(len(eval_idx), cap, replace=False))
        ev = eval_idx[pick]
        b.n_eval_used = int(cap)
    for kind, rg in [("base", reg)] + sorted(grown.items()):
        d = {"M": rg.M, "n_coll": rg.n_collections()}
        R = E[rg.idx]
        d["Z_sup"] = C.max_cos(E[sup_idx], R)
        d["Z_eval"] = C.max_cos(E[ev], R)
        d["hist_sup"] = {}
        if role == "dev" and kind == "base":
            Rm = R if len(R) <= 200 else R[np.sort(np.random.default_rng(seed + 555).choice(len(R), 200, replace=False))]
            d["hist_sup"] = {100: (np.asarray(E[sup_idx[:100]], np.float32) @ np.asarray(Rm, np.float32).T).ravel().astype(np.float64)}
        if pv is not None and len(pv):
            d["Z_pos"] = C.max_cos(Ev, R)
            d["pos_variant"] = pv.variant.to_numpy()
            d["n_pos_orig"] = int(pv.file.nunique())
        else:
            d["Z_pos"] = np.zeros(0)
            d["pos_variant"] = np.zeros(0, dtype=object)
            d["n_pos_orig"] = 0
        b.reg[kind] = d
    return b


# (dataset, file, variant) -> SSCD descriptor
class Lookup:

    def __init__(self, smoke, dim=512):
        self.synthetic = smoke
        self.dim = dim
        self.rows, self.mats = ({}, {})
        if smoke:
            return
        for name in ("solana", "nft", "disc21_bg"):
            idx, emb = C.EmbStore.open_read(C.EMB / name)
            self.mats[name] = emb
            self.rows.update({k: (name, i) for i, k in enumerate(zip(idx.dataset, idx.file, idx.variant))})

    def _get(self, keys):
        out = np.empty((len(keys), self.dim), np.float32)
        if self.synthetic:
            import hashlib
            for j, k in enumerate(keys):
                r = np.random.default_rng(int(hashlib.sha256(repr(k).encode()).hexdigest()[:8], 16)).normal(size=self.dim)
                out[j] = r / np.linalg.norm(r)
            return out
        by = {}
        for j, k in enumerate(keys):
            if k not in self.rows:
                raise KeyError(f"no SSCD descriptor for {k}")
            name, i = self.rows[k]
            by.setdefault(name, ([], []))[0].append(j)
            by[name][1].append(i)
        for name, (js, iis) in by.items():
            iis = np.asarray(iis)
            order = np.argsort(iis)
            out[np.asarray(js)[order]] = self.mats[name][iis[order]]
        return out

    def orig(self, df):
        return self._get(list(zip(df.dataset, df.file, ["orig"] * len(df))))

    def orig_files(self, pv):
        return self._get(list(zip(pv.dataset, pv.file, ["orig"] * len(pv))))

    def var(self, pv):
        return self._get(list(zip(pv.dataset, pv.file, pv.variant)))


def main(args, out):
    S, MF, T, REG = C.modules()

    _, m, cfg = S.X.reference_inputs(S.X.DEFAULT)
    dbp, _ = S.X.load_tasks(m)
    ref = {(b.role, b.domain, b.structure, int(b.M), int(b.seed)): b for b in dbp if b.scheme == "dbp"}
    keys = sorted(ref)
    if args.smoke:
        keys = [k for r in ("train", "dev", "test") for k in [x for x in keys if x[0] == r][:3]]

    root = Path(cfg["root"])
    table = T.load_role_table(m["role_table"])
    pools, variants = ({}, {})
    for name, pc in cfg["pools"].items():
        pools[name] = REG.Pool(root / pc["cache"], name=name, scheme="dbp")
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
        dom_pool[dom] = pools[names[0]] if len(names) == 1 else REG.Pool([root / cfg["pools"][n]["cache"] for n in names], name=dom, scheme="dbp")
        dom_var[dom] = pd.concat([variants[n] for n in names if n in variants], ignore_index=True) if any(n in variants for n in names) else None
        dom_roles[dom] = T.roles_from_table(dom_pool[dom], table)
    info = json.loads((C.WEIGHTS.parent / "weights_info.json").read_text(encoding="utf-8")) if C.WEIGHTS.parent.joinpath("weights_info.json").exists() else {"dim": 512}
    lookup = Lookup(args.smoke, info["dim"])

    folder = C.output_path(C.OUT_ROOT / "data" / "sscd_tasks_smoke" if args.smoke else C.ROOT / "data" / "sscd_tasks")
    folder.mkdir(parents=True, exist_ok=True)
    rp_cache = {}
    for i, key in enumerate(keys):
        role, dom, struct, M, seed = key
        tok = task_token(["sscd"] + list(key))
        npz = folder / (tok + ".npz")
        rec = folder / (tok + ".complete.json")
        if rec.exists():

            continue
        if (dom, role) not in rp_cache:
            rp = T.role_pool(dom_pool[dom], dom_roles[dom], role)
            rp_cache[dom, role] = (rp, lookup.orig(rp.df))
        rp, E = rp_cache[dom, role]
        b = build_one(T, rp, dom_var[dom], role, dom, struct, M, seed, cfg, lookup, E)
        if b.note:
            raise RuntimeError(f"task could not be built for {key}: {b.note}")
        r = ref[key]
        same = list(map(str, b.sup_files)) == list(map(str, r.sup_files)) and list(map(str, b.eval_files)) == list(map(str, r.eval_files)) and (set(b.reg) == set(r.reg)) and (list(map(str, b.gen_variant)) == list(map(str, r.gen_variant))) and all((list(map(str, b.reg[k]["pos_variant"])) == list(map(str, r.reg[k]["pos_variant"])) for k in b.reg))
        if not same:
            raise RuntimeError(f"SSCD task sampled different files or positives than DBpHash for {key}")
        tmp = folder / (tok + ".pending.npz")
        T.save_bundle(b, tmp)
        Path(str(tmp)).replace(npz)
        Path(str(tmp) + ".json").replace(Path(str(npz) + ".json"))
        C.atomic_json(
            rec,
            {
                "scheme": "sscd",
                "key": list(key),
                "npz_sha256": C.sha256(npz),
                "json_sha256": C.sha256(str(npz) + ".json"),
                "same_files_as_dbphash": True,
                "time": datetime.datetime.now().isoformat(),
            },
        )

    return {"scheme": "sscd", "tasks": len(keys), "cache": str(folder)}
if __name__ == "__main__":
    C.run_step("sscd_build", "Build SSCD tasks", main)
