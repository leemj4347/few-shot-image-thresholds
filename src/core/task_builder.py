# Reference model comparison
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
import progress as RP
import progress_summary as OP
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


# Cache only locally generated bundles; verify every completed entry
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
            RP.update("task_cache", domain=domain, role=role, M=int(M), seed=int(seed), cache_hit=True)
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


def _run(config_path, role_path, out, inclusion=None, build_only=False):
    out = Path(out)
    if out.exists():
        raise ValueError("output directory already exists")
    cfg = read_json(config_path)
    if cfg["meta_quantile"]["epochs"] != 15 or cfg["meta_quantile"]["hidden"] != 256:
        raise ValueError("Primary reference config requires 15 epochs and hidden 256")
    if cfg.get("schemes") != ["dbp"]:
        raise ValueError("This reference run is DBpHash only")
    if cfg.get("n_max") != 100:
        raise ValueError("Primary support is fixed at 100")
    if cfg.get("train_M") != [10, 100, 1000]:
        raise ValueError("Training/calibration M scope changed")
    out.mkdir(parents=True)
    cfg.setdefault("support_random", True)
    cfg.setdefault("eval_all", True)
    if not cfg["support_random"] or not cfg["eval_all"]:
        raise ValueError("Required sampling protocol changed")
    cfg["control_hidden"] = 256
    cfg["control_epochs"] = 15
    cfg["control_seed"] = 42
    role_path = Path(role_path)
    if inclusion is not None and (not Path(inclusion).is_file()):
        raise FileNotFoundError(inclusion)
    if inclusion is None and cfg["meta_quantile"].get("requires_inclusion", True):
        raise ValueError("inclusion file required")
    srcfiles = [HERE / p for p in ("tasks.py", "registry.py", "similarity.py", "transforms.py", "evaluate_registry.py")]
    codefiles = srcfiles + [
        Path(__file__),
        HERE / "methods" / "meta_quantile.py",
        HERE / "methods" / "calibration.py",
        HERE / "registry_evaluation.py",
        HERE / "criteria.py",
        HERE / "progress_summary.py",
    ]
    hashes = {str(p): sha(p) for p in codefiles}
    geometry = {k: v for k, v in cfg.items() if k not in ("meta_quantile", "control_hidden", "control_epochs", "control_seed")}
    overall = OP.Overall(out, cfg, build_only)
    inputs = {}
    for pool in cfg.get("pools", {}).values():
        for field in ("cache", "variants", "variants2"):
            if pool.get(field):
                path = Path(cfg["root"]) / pool[field]
                if not path.is_file():
                    raise FileNotFoundError(path)
                inputs[str(path)] = sha(path)
                overall.advance("prepare", path=str(path))
    cache_spec = {
        "geometry": geometry,
        "role_sha256": sha(role_path),
        "input_sha256": inputs,
        "sources": {str(p): sha(p) for p in srcfiles},
    }
    fp = hashlib.sha256(json.dumps(cache_spec, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    cache = Path(cfg["root"]) / "data" / "registry_tasks" / fp
    manifest = {
        "status": "started",
        "config": cfg,
        "source_commit": F.COMMIT,
        "role_table": str(role_path),
        "role_sha256": sha(role_path),
        "inclusion": str(inclusion) if inclusion else None,
        "inclusion_sha256": sha(inclusion) if inclusion else None,
        "code_sha256": hashes,
        "input_sha256": inputs,
        "task_cache_fingerprint": fp,
        "task_cache": str(cache),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    (out / "config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "reference_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    original_build = T.build_bundle
    original_backend = R.MS
    original_n = T.N_LIST[:]
    original_update = RP.update

    def observe(stage, **fields):
        overall.event(stage, fields)
        original_update(stage, **fields)
    RP.update = observe
    try:
        T.N_LIST = [100]
        T.build_bundle = task_cache(original_build, cache, fp)
        with RP.monitor(out):
            table = T.load_role_table(role_path)
            invalid = []
            bundles = R.build_bundles(cfg, Path(cfg["root"]), table, "dbp", invalid=invalid)
            overall.close("pool_load", skipped=True)
            overall.close("task_build", skipped=True)
            (out / "invalid_tasks.json").write_text(json.dumps(invalid, ensure_ascii=False, indent=2), encoding="utf-8")
            if build_only:
                manifest["status"] = "task_build_completed_no_training"
                manifest["bundle_count"] = len(bundles)
                overall.state("build_only_completed")
            else:
                included = R.load_included(Path(inclusion) if inclusion else None)
                validation = [b for b in bundles if b.role == "dev" and b.M in cfg["train_M"]]
                if not validation:
                    raise ValueError("No in-range development validation tasks")
                fitb, calb = R.split_train_cal([b for b in bundles if b.role == "train"], cfg.get("cal_frac", 0.5))
                if not fitb or not calb:
                    raise ValueError("Empty fit/calibration split")
                manifest["fit_task_count"] = len(fitb)
                manifest["calibration_task_count"] = len(calb)
                actual_tests = sum(b.role == "test" for b in bundles)
                planned_tests = overall.data["buckets"]["evaluation"]["total"] // 2
                overall.advance("evaluation", 2 * max(0, planned_tests - actual_tests), skipped=True)
                frames = []
                infos = {}
                for mode, label in (("summary", "요약 통계량 MLP"), ("deepset", "Fisch DeepSet 원코드")):
                    overall.data["model"] = mode
                    RP.update("model_start", model=mode)
                    R.MS = F.Backend(mode, validation, out / "models" / mode)
                    R.MS.overall = overall
                    df, info = R.score_all(bundles, cfg, "dbp", included)
                    mask = df.method == "요약 통계량 MLP"
                    df.loc[mask, "method"] = label
                    frames.append(df if mode == "summary" else df[df.method == label])
                    infos[mode] = info
                data = pd.concat(frames, ignore_index=True)
                if data.empty:
                    raise ValueError("No evaluation rows")
                missing = R.missing_included_columns(data, included)
                if missing:
                    raise RuntimeError("Missing expected variant columns: " + str(missing))
                RP.update("aggregate")
                conditions = R.condition_table(data)
                overall.advance("aggregate")
                mc = R.mcnemar_table(data)
                overall.advance("aggregate")
                gate = R.gate(conditions)
                overall.advance("aggregate")
                RP.update("save")
                data.to_csv(out / "registry_rows.csv", index=False, encoding="utf-8-sig")
                conditions.to_csv(out / "registry_conditions.csv", index=False, encoding="utf-8-sig")
                mc.to_csv(out / "registry_mcnemar.csv", index=False, encoding="utf-8-sig")
                (out / "registry_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
                manifest.update(status="completed", model_info=infos, n_rows=len(data), n_conditions=len(conditions))
                overall.advance("save")
                overall.state("awaiting_verification")
    except BaseException as exc:
        manifest["status"] = "interrupted_or_failed"
        manifest["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        T.build_bundle = original_build
        T.N_LIST = original_n
        R.MS = original_backend
        RP.update = original_update
        (out / "reference_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def run(config_path, role_path, out, inclusion=None, build_only=False):
    try:
        return _run(config_path, role_path, out, inclusion, build_only)
    except BaseException:
        if (Path(out) / "progress_summary.json").exists():
            try:
                OP.finalize(out, failed=True)
            except OSError:
                pass
        raise
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--role-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inclusion")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    run(args.config, args.role_table, args.out, args.inclusion, args.build_only)
