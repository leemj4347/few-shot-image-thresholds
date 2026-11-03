# Load task manifests and cached bundles
from pathlib import Path
import sys, json, hashlib, time, datetime, threading, os, copy, argparse
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tasks as T
import criteria as CR
import evaluate_registry as EV
import registry_evaluation as R
from methods import registry_predictor as F
from task_builder import sha, read_json
DEFAULT = Path(__file__).resolve().parents[2] / "results/tasks/reference"
KEYS = R.COND_KEYS + ["scheme"]


def atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    for k in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == 19:
                raise
            time.sleep(0.05)


def reference_inputs(reference):
    reference = Path(reference)
    m = read_json(reference / "reference_manifest.json")
    if m["status"] not in ("completed", "task_build_completed_no_training"):
        raise ValueError("Reference run not complete")
    F.source_class()
    for path, digest in m["code_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Reference source changed: " + path)
    for path, digest in m["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Reference input changed: " + path)
    if sha(m["role_table"]) != m["role_sha256"]:
        raise ValueError("Role table changed")
    if sha(m["inclusion"]) != m["inclusion_sha256"]:
        raise ValueError("Inclusion changed")
    if m["config"]["train_M"] != [10, 100, 1000]:
        raise ValueError("Training scope changed")
    cfg = copy.deepcopy(m["config"])
    cfg.pop("gate", None)
    cfg["control_epochs"] = 15
    cfg["control_seed"] = 42
    return (reference, m, cfg)


def receipts(m):
    return sorted(Path(m["task_cache"]).glob("*.complete.json"))


def load_tasks(m, progress=None):
    bundles = []
    invalid = []
    for receipt in receipts(m):
        rec = read_json(receipt)
        token = receipt.name.removesuffix(".complete.json")
        p = receipt.parent / (token + ".npz")
        meta_path = Path(str(p) + ".json")
        if rec["fingerprint"] != m["task_cache_fingerprint"]:
            raise ValueError("Task fingerprint mismatch")
        meta = read_json(meta_path)
        expected_key = [meta.get(k) for k in ("scheme", "role", "domain", "structure", "M", "seed")]
        if rec["key"][1:] != expected_key:
            raise ValueError(f"cached bundle {token} does not match its metadata key")
        if sha(p) != rec["npz_sha256"] or sha(meta_path) != rec["json_sha256"]:
            raise ValueError("Cached task checksum mismatch")
        valid = meta.get("n_sup", 0) >= 100 and meta.get("n_eval", 0) > 0 and ("base" in meta.get("kinds", {}))
        if valid:
            b = T.load_bundle(p)
            for name in T._SCALARS:
                if name in meta:
                    setattr(b, name, meta[name])
            bundles.append(b)
        else:
            invalid.append(meta)
        if progress:
            progress.data["invalid_tasks"] = len(invalid)
            progress.advance(
                "task_cache",
                skipped=not valid,
                role=meta.get("role"),
                domain=meta.get("domain"),
                M=meta.get("M"),
            )
    if len(set((F.key(b) for b in bundles))) != len(bundles):
        raise ValueError("Duplicate cache tasks")
    return (bundles, invalid)
