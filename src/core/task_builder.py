from __future__ import annotations

import argparse

import hashlib

import json

import sys

import platform

import importlib.metadata

from pathlib import Path

import numpy as np

import pandas as pd

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))

sys.path.insert(0, str(HERE))

import tasks as T

import registry_evaluation as R

from methods import meta_quantile as F

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def task_cache(original, folder, fingerprint):
    folder.mkdir(parents=True, exist_ok=True)

    def build(pool, var, role, domain, structure, M, seed, cfg):
        key = [pool.name, pool.scheme, role, domain, structure, int(M), int(seed)]
        token = hashlib.sha256(json.dumps(key).encode()).hexdigest()
        p = folder / (token + ".npz")
        receipt = folder / (token + ".complete.json")
        if receipt.exists():
            rec = read_json(receipt)
            if rec["fingerprint"] != fingerprint or rec["key"] != key:
                raise RuntimeError(f"cached bundle {token} was built from a different fingerprint or key")
            if sha(p) != rec["npz_sha256"] or sha(str(p) + ".json") != rec["json_sha256"]:
                raise RuntimeError("Task cache checksum mismatch")
            b = T.load_bundle(p)
            meta = read_json(str(p) + ".json")
            for field in T._SCALARS:
                if field in meta:
                    setattr(b, field, meta[field])

            return b
        b = original(pool, var, role, domain, structure, M, seed, cfg)
        tmp = folder / (token + ".pending.npz")
        T.save_bundle(b, tmp)
        tmp.replace(p)
        Path(str(tmp) + ".json").replace(Path(str(p) + ".json"))
        rec = {"fingerprint": fingerprint, "key": key, "npz_sha256": sha(p), "json_sha256": sha(str(p) + ".json")}
        receipt.write_text(json.dumps(rec), encoding="utf-8")
        return b
    return build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--role-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inclusion")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    run(args.config, args.role_table, args.out, args.inclusion, args.build_only)

def run(config_path, role_path, out, inclusion=None, build_only=True):
    if not build_only:
        raise ValueError("Use the predictor training stage after task construction.")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    cfg = read_json(config_path)
    if cfg["meta_quantile"]["epochs"] != 15 or cfg["meta_quantile"]["hidden"] != 256:
        raise ValueError("Expected 15 epochs and hidden width 256.")
    if cfg.get("schemes") != ["dbp"] or cfg.get("n_max") != 100 or cfg.get("train_M") != [10, 100, 1000]:
        raise ValueError("Unexpected task settings.")
    cfg.setdefault("support_random", True)
    cfg.setdefault("eval_all", True)
    if not cfg["support_random"] or not cfg["eval_all"]:
        raise ValueError("Random support sampling and all evaluation queries are required.")
    cfg.update(control_hidden=256, control_epochs=15, control_seed=42)
    role_path = Path(role_path)
    if inclusion is None or not Path(inclusion).is_file():
        raise FileNotFoundError(inclusion)
    srcfiles = [HERE / name for name in ["tasks.py", "registry.py", "similarity.py", "transforms.py", "evaluate_registry.py"]]
    codefiles = srcfiles + [Path(__file__), HERE / "methods/meta_quantile.py", HERE / "methods/calibration.py", HERE / "registry_evaluation.py", HERE / "criteria.py"]
    inputs = {}
    for pool in cfg["pools"].values():
        for field in ["cache", "variants", "variants2"]:
            if pool.get(field):
                path = Path(cfg["root"]) / pool[field]
                inputs[str(path)] = sha(path)
    geometry = {k: v for k, v in cfg.items() if k not in ["meta_quantile", "control_hidden", "control_epochs", "control_seed"]}
    spec = {"geometry": geometry, "role_sha256": sha(role_path), "input_sha256": inputs, "sources": {str(p): sha(p) for p in srcfiles}}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    cache = Path(cfg["root"]) / "data/registry_tasks" / fingerprint
    manifest = {"status": "started", "config": cfg, "source_commit": F.COMMIT,
                "role_table": str(role_path), "role_sha256": sha(role_path),
                "inclusion": str(inclusion), "inclusion_sha256": sha(inclusion),
                "code_sha256": {str(p): sha(p) for p in codefiles}, "input_sha256": inputs,
                "task_cache_fingerprint": fingerprint, "task_cache": str(cache)}
    out.mkdir(parents=True)
    original_build, original_n = T.build_bundle, T.N_LIST[:]
    try:
        T.N_LIST = [100]
        T.build_bundle = task_cache(original_build, cache, fingerprint)
        invalid = []
        bundles = R.build_bundles(cfg, Path(cfg["root"]), T.load_role_table(role_path), "dbp", log=lambda *a: None, invalid=invalid)
        (out / "invalid_tasks.json").write_text(json.dumps(invalid, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.update(status="task_build_completed_no_training", bundle_count=len(bundles))
    finally:
        T.build_bundle, T.N_LIST = original_build, original_n
        (out / "reference_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
