# Build registry tasks from reference sampling settings
from __future__ import annotations
import json, datetime
from pathlib import Path
import numpy as np
import experiment as C

TASK_REFERENCE = C.INPUTS / "task_reference"


def config_path():
    m = json.loads((TASK_REFERENCE / "reference_manifest.json").read_text(encoding="utf-8-sig"))
    cfg = dict(m["config"])
    cfg["root"] = str(C.ROOT).replace("\\", "/")
    if "fisch_reference" in cfg:
        cfg["meta_quantile"] = cfg.pop("fisch_reference")
    for pool in cfg.get("pools", {}).values():
        for key in ("cache", "variants", "variants2"):
            if pool.get(key):
                pool[key] = "data/hashes/" + Path(pool[key]).name
    d = C.output_path(C.OUT_ROOT / "data" / "design")
    d.mkdir(parents=True, exist_ok=True)
    p = d / "config_reference.json"
    C.atomic_json(p, cfg)
    return p


def main(args, out):
    cfgp = config_path()
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    role = C.ROOT / "data" / "hashes" / "role_table.csv"
    incl = C.ROOT / "data" / "design" / "inclusion_fixed.json"
    for pth in [role, incl] + [Path(cfg["root"]) / pc[k] for pc in cfg["pools"].values() for k in ("cache", "variants", "variants2") if pc.get(k)]:
        if not Path(pth).exists(): raise FileNotFoundError(f"run the hashes step first: {pth}")
    import pandas as pd
    for pc in cfg["pools"].values():
        d = pd.read_parquet(Path(cfg["root"]) / pc["cache"], columns=["mid_u64", "high_u64", "phash_u64"])
        if not all(d[c].dtype == np.uint64 for c in d): raise TypeError("hash cache is not uint64")
    if args.smoke:
        return None
    C.runner()
    import task_builder as RF
    ref = C.REFERENCE
    status = json.loads((ref / "reference_manifest.json").read_text(encoding="utf-8")).get("status") if (ref / "reference_manifest.json").exists() else None
    if status != "task_build_completed_no_training":
        if ref.exists():
            ref.rename(ref.with_name(f"reference_incomplete_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
        RF.run(str(cfgp), str(role), str(ref), str(incl), True)
    m = json.loads((ref / "reference_manifest.json").read_text(encoding="utf-8"))
    if m.get("status") != "task_build_completed_no_training": raise RuntimeError(f"task build status {m.get('status')}")
    C.load_reference()


if __name__ == "__main__":
    C.run_step("tasks", "Build tasks", main)
