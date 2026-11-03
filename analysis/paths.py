from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("EXPERIMENT_RESULTS", str(ROOT / "results"))).resolve()


def output_root():
    p = Path(os.environ["ANALYSIS_OUTPUT"]).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def task_cache(manifest):
    local = ROOT / "data/registry_tasks" / manifest["task_cache_fingerprint"]
    if local.is_dir():
        return local
    raise FileNotFoundError("task cache missing: " + str(local))
