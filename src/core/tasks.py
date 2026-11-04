# Role splits and per-task score bundles
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

import similarity as SIM
from registry import Pool, Registry, sample_queries, positives_for, holdout_collections, variant_hashes, reserve_collections

N_LIST = [10, 20, 50, 100]
N_MAX = max(N_LIST)
N_EVAL = 299
ROLES = ("train", "dev", "test")


def _stable_bucket(key: str, salt: str, k: int) -> int:
    return int(hashlib.sha256((salt + "|" + str(key)).encode("utf-8")).hexdigest()[:8], 16) % k


# Unit of the role split
def split_keys(pool: Pool) -> np.ndarray:
    coll = pool.coll.astype(object)
    gid = pool.df.group_id.to_numpy()
    return np.array([c if c else ("__grp__" + str(g)) for c, g in zip(coll, gid)], dtype=object)


ROLE_SHARE = {"train": 0.4, "dev": 0.3, "test": 0.3}


# Split meta-training/development/test 4:3:3 by collection (pixel group if absent)
def role_split(pool: Pool, salt: str = "phase1-role-2026") -> dict:
    keys, counts = np.unique(split_keys(pool), return_counts=True)
    h = {k: hashlib.sha256((salt + "|" + str(k)).encode("utf-8")).hexdigest() for k in keys}
    order = sorted(zip(keys, counts), key=lambda kc: (-int(kc[1]), h[kc[0]]))
    out = {r: set() for r in ROLES}
    size = {r: 0 for r in ROLES}
    for k, c in order:
        r = min(ROLES, key=lambda r: (size[r] / ROLE_SHARE[r], ROLES.index(r)))
        out[r].add(k)
        size[r] += int(c)
    return out


# Pool restricted to one role
def role_pool(pool: Pool, roles: dict, role: str) -> Pool:
    keep = roles[role]
    sub = pool.df[pd.Series(split_keys(pool), index=pool.df.index).isin(keep)]
    p = object.__new__(Pool)
    p.df = sub.reset_index(drop=True)
    p.name = f"{pool.name}/{role}"
    p.scheme = pool.scheme
    p.w = pool.w
    idx = sub.index.to_numpy()
    p.mid = pool.mid[idx]
    p.high = pool.high[idx]
    p.coll = pool.coll[idx]
    p.has_coll = bool((p.coll != "").any())
    p.n = len(sub)
    return p


@dataclass
class Bundle:
    role: str
    domain: str
    structure: str
    M: int
    seed: int
    scheme: str
    n_coll_registry: int = 0
    n_var_members: int = 0
    n_eval_used: int = 0
    n_eval: int = 0
    n_sup: int = 0
    reg: dict = field(default_factory=dict)
    sup_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    eval_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    sup_files: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    eval_files: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    hold: set = field(default_factory=set)
    reserve: set = field(default_factory=set)
    S_gen: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gen_variant: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    note: str = ""

    @property
    def w(self): return SIM.SCHEME_W[self.scheme]


# Cumulative 65x65 pair counts over the first n support queries
def _hists(pool_w, qm, qh, rm, rh, n_list):
    out = {}
    prev = np.zeros((65, 65), dtype=np.int64)
    start = 0
    for n in n_list:
        if n > len(qm): break
        r = SIM.query_max(qm[start:n], qh[start:n], rm, rh, w=pool_w, hist=True)
        prev = prev + r["hist"]
        out[n] = prev.copy()
        start = n
    return out


# Score bundle for one task
def build_bundle(pool: Pool, variants: pd.DataFrame | None, role: str, domain: str, structure: str,
                 M: int, seed: int, cfg: dict) -> Bundle | None:
    b = Bundle(role=role, domain=domain, structure=structure, M=M, seed=seed, scheme=pool.scheme)
    minr = cfg.get("holdout_min_reps", 50)
    need = cfg.get("query_holdout_need")
    hold = holdout_collections(
        pool,
        seed,
        cfg.get("query_collection_frac", 0.3),
        minr,
        need,
    ) if pool.has_coll else set()
    gf = cfg.get("growth", {}).get("factor", 10)
    nc = cfg.get("growth", {}).get("new_collections", 3)
    reserve = reserve_collections(pool, seed, nc, hold, minr)
    try:
        reg = Registry.sample(pool, M, seed, structure, exclude_coll=hold | reserve)
    except ValueError as e:
        b.note = f"registry: {e}"
        return b


    n_var_members = 0
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
            n_var_members = int(pool.df.file.isin(full).to_numpy()[reg.idx].sum())
    grown = {}
    try: grown["grow_i"] = reg.grow(M * gf)
    except ValueError: pass
    if reserve:
        try: grown["grow_ii"] = reg.add_collections(nc, seed, per_coll=cfg.get("growth", {}).get("per_collection"), allowed=reserve)
        except ValueError: pass

    allidx = np.unique(np.concatenate([reg.idx] + [g.idx for g in grown.values()])) if grown else reg.idx
    tmp = Registry(pool, allidx, structure, seed, 0, exclude_coll=hold)


    want = pool.n if cfg.get("eval_all") else N_MAX + N_EVAL
    q = sample_queries(pool, tmp, want, seed, exclude_files=cfg.get("_exclude_files") or None)
    if len(q) < cfg.get("n_min_total", 120):
        b.note = f"queries {len(q)} < min"
        return b
    if cfg.get("support_random"):
        q = q[np.random.default_rng(seed + 90001).permutation(len(q))]
    n_sup = min(N_MAX, len(q) - 1) if cfg.get("eval_all") else min(N_MAX, len(q) // 3)
    sup_idx, eval_idx = q[:n_sup], q[n_sup:]
    b.n_sup = len(sup_idx)
    b.n_eval = len(eval_idx)
    b.n_coll_registry = reg.n_collections()
    b.n_var_members = n_var_members
    b.sup_idx = sup_idx
    b.eval_idx = eval_idx
    b.hold = set(hold)
    b.reserve = set(reserve)
    b.sup_files = pool.df.file.to_numpy()[sup_idx]
    b.eval_files = pool.df.file.to_numpy()[eval_idx]
    qsm, qsh = pool.subset(sup_idx)
    qem, qeh = pool.subset(eval_idx)
    n_list = [n for n in N_LIST if n <= b.n_sup]
    pv = positives_for(reg, variants, cfg.get("n_pos_originals", 200), seed) if variants is not None else None
    if pv is not None and len(pv):
        pvm, pvh = variant_hashes(pool, pv)
        b.S_gen = _genuine_scores(pool, pv, pvm, pvh)
        b.gen_variant = pv.variant.to_numpy()


    cap = int(cfg.get("train_eval_cap", 0)) if role == "train" else 0
    if cap and len(eval_idx) > cap:
        pick = np.sort(np.random.default_rng(seed + 7777).choice(len(eval_idx), cap, replace=False))
        qem, qeh = pool.subset(eval_idx[pick])
        b.n_eval_used = int(cap)
    for kind, rg in [("base", reg)] + sorted(grown.items()):
        d = {"M": rg.M, "n_coll": rg.n_collections()}
        d["Z_sup"] = rg.query_max(qsm, qsh)["Z"]
        d["Z_eval"] = rg.query_max(qem, qeh)["Z"]
        d["hist_sup"] = _hists(pool.w, qsm, qsh, rg.mid, rg.high, n_list)
        if pv is not None and len(pv):
            pvm, pvh = variant_hashes(pool, pv)
            d["Z_pos"] = rg.query_max(pvm, pvh)["Z"]
            d["pos_variant"] = pv.variant.to_numpy()
            d["n_pos_orig"] = int(pv.file.nunique())
        else:
            d["Z_pos"] = np.zeros(0)
            d["pos_variant"] = np.zeros(0, dtype=object)
            d["n_pos_orig"] = 0
        b.reg[kind] = d
    return b


# Pair scores S(variant, original)
def _genuine_scores(pool: Pool, pv: pd.DataFrame, pvm, pvh) -> np.ndarray:
    by = pool.df.set_index("file")
    rows = by.loc[pv.file.tolist()]
    from registry import SCHEMES
    mcol, hcol, _ = SCHEMES[pool.scheme]
    om = rows[mcol].to_numpy(np.uint64)
    oh = rows[hcol].to_numpy(np.uint64) if hcol else np.zeros(len(rows), np.uint64)
    dm = SIM.popcount_u64(pvm ^ om)
    dh = SIM.popcount_u64(pvh ^ oh)
    return SIM.sim_from_dist(dm, dh, pool.w)


_SCALARS = ("role", "domain", "structure", "M", "seed", "scheme", "n_coll_registry", "n_eval", "n_sup", "note")


def save_bundle(b: Bundle, path: Path):
    arr = {"sup_files": b.sup_files.astype(object), "eval_files": b.eval_files.astype(object),
           "sup_idx": b.sup_idx, "eval_idx": b.eval_idx, "S_gen": b.S_gen, "gen_variant": b.gen_variant.astype(object)}
    meta = {k: getattr(b, k) for k in _SCALARS}
    meta["kinds"] = {}
    for kind, d in b.reg.items():
        meta["kinds"][kind] = {"M": int(d["M"]), "n_coll": int(d["n_coll"]), "n_pos_orig": int(d["n_pos_orig"]),
                               "hist_n": sorted(int(n) for n in d["hist_sup"])}
        arr[f"{kind}|Z_sup"] = d["Z_sup"]
        arr[f"{kind}|Z_eval"] = d["Z_eval"]
        arr[f"{kind}|Z_pos"] = d["Z_pos"]
        arr[f"{kind}|pos_variant"] = d["pos_variant"].astype(object)
        for n, H in d["hist_sup"].items(): arr[f"{kind}|hist|{n}"] = H
    meta["hold"] = sorted(b.hold)
    meta["reserve"] = sorted(b.reserve)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arr)
    import json as _json
    open(str(path) + ".json", "w", encoding="utf-8").write(_json.dumps(meta, ensure_ascii=False, default=str))


def load_bundle(path: Path) -> Bundle:
    import json as _json
    meta = _json.load(open(str(path) + ".json", encoding="utf-8"))
    z = np.load(path, allow_pickle=True)
    b = Bundle(**{k: meta[k] for k in ("role", "domain", "structure", "scheme", "note")},
               M=int(meta["M"]), seed=int(meta["seed"]))
    b.n_coll_registry = meta["n_coll_registry"]
    b.n_eval = meta["n_eval"]
    b.n_sup = meta["n_sup"]
    b.sup_files = z["sup_files"]
    b.eval_files = z["eval_files"]
    b.sup_idx = z["sup_idx"]
    b.eval_idx = z["eval_idx"]
    b.S_gen = z["S_gen"]
    b.gen_variant = z["gen_variant"]
    b.hold = set(meta.get("hold", []))
    b.reserve = set(meta.get("reserve", []))
    for kind, km in meta["kinds"].items():
        b.reg[kind] = {"M": km["M"], "n_coll": km["n_coll"], "n_pos_orig": km["n_pos_orig"],
                       "Z_sup": z[f"{kind}|Z_sup"], "Z_eval": z[f"{kind}|Z_eval"],
                       "Z_pos": z[f"{kind}|Z_pos"], "pos_variant": z[f"{kind}|pos_variant"],
                       "hist_sup": {int(n): z[f"{kind}|hist|{n}"] for n in km["hist_n"]}}
    return b


# Bundle summary (diagnostic)
def bundle_rows(b: Bundle) -> list[dict]:
    out = []
    for kind, d in b.reg.items():
        out.append({"role": b.role, "domain": b.domain, "structure": b.structure, "M": b.M, "seed": b.seed,
                    "scheme": b.scheme, "kind": kind, "M_eff": d["M"], "n_coll": d["n_coll"],
                    "n_sup": b.n_sup, "n_eval": b.n_eval, "n_pos_orig": d["n_pos_orig"],
                    "Z_eval_max": float(d["Z_eval"].max()) if len(d["Z_eval"]) else np.nan,
                    "Z_sup_max": float(d["Z_sup"].max()) if len(d["Z_sup"]) else np.nan})
    return out


# Role table -> {key: role}
def load_role_table(path) -> dict:
    rt = pd.read_csv(path, dtype={"key": str})
    return dict(zip(rt.key.astype(str), rt.role.astype(str)))


# Split this pool's keys by role using the role table
def roles_from_table(pool: Pool, table: dict) -> dict:
    keys = np.unique(split_keys(pool))
    missing = [k for k in keys if k not in table]
    if missing:
        raise RuntimeError(f"Role table missing {len(missing)} keys (e.g. {missing[:3]})")
    out = {r: set() for r in ROLES}
    for k in keys:
        out[table[k]].add(k)
    return out
