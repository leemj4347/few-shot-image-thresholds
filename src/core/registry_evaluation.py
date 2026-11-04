# Construct registry task bundles
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tasks as T
from registry import Pool

COND_KEYS = ["domain", "structure", "kind", "M", "alpha"]


def build_bundles(cfg: dict, root: Path, role_table: dict, scheme: str, log=print,
                  invalid: list | None = None) -> list:
    if invalid is None: invalid = []
    pools, variants = {}, {}
    for pool_index, (name, p) in enumerate(cfg["pools"].items(), 1):

        pools[name] = Pool(root / p["cache"], name=name, scheme=scheme)
        frames = []
        for key in ("variants", "variants2"):
            rel = p.get(key)
            if rel is None and key == "variants2" and p.get("variants"):
                rel = str(p["variants"]).replace("_var.parquet", "_var2.parquet")
            if rel and (root / rel).exists():
                v = pd.read_parquet(root / rel)
                frames.append(v[v.ok & (v.variant != "orig")])
        if frames: variants[name] = pd.concat(frames, ignore_index=True)

    out = []
    for dom, d in cfg["domains"].items():
        names = d["pools"]
        pool = pools[names[0]] if len(names) == 1 else Pool([root / cfg["pools"][n]["cache"] for n in names],
                                                           name=dom, scheme=scheme)
        var = pd.concat([variants[n] for n in names if n in variants], ignore_index=True) \
            if any(n in variants for n in names) else None
        roles = T.roles_from_table(pool, role_table)
        for role in T.ROLES:

            if role == "train" and scheme != "dbp": continue
            rp = T.role_pool(pool, roles, role)
            if rp.n < 50: continue

            M_list = cfg.get("train_M", d["M"]) if role == "train" else d["M"]
            structures=[st for st in d.get("structures", ["random"]) if st=="random" or rp.has_coll]
            for struct in structures:
                for M in M_list:
                    for seed in cfg["seeds"]:

                        b = T.build_bundle(rp, var, role, dom, struct, M, seed, cfg)

                        if b.note:
                            invalid.append({"role": role, "domain": dom, "structure": struct,
                                            "M": int(M), "seed": int(seed), "reason": b.note})
                            continue
                        out.append(b)
        n_bad = sum(1 for r in invalid if r["domain"] == dom)
        log(f"  {dom}: {sum(1 for b in out if b.domain == dom)} tasks, {n_bad} invalid")
    return out


def split_train_cal(train_bundles: list, frac: float = 0.7) -> tuple[list, list]:
    seeds = sorted({b.seed for b in train_bundles})
    cut = max(1, int(round(len(seeds) * frac)))
    fit_seeds = set(seeds[:cut])
    cal_seeds = set(seeds[cut:])
    if not cal_seeds or fit_seeds & cal_seeds: raise ValueError("disjoint fit/cal seeds required")
    return ([b for b in train_bundles if b.seed in fit_seeds],
            [b for b in train_bundles if b.seed in cal_seeds])
