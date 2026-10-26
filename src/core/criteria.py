# Condition-level risk and balanced-accuracy criteria
from __future__ import annotations
import numpy as np
import similarity as SIM


DELTA = 0.05
BASELINE_METHOD = "원문 식 28"


# Secondary edit names used for detection rate
def variant_names() -> list[str]:
    import transforms as TR
    return [s[0] for s in TR.SPEC_V3]


# False alarm rate P(Z >= tau) at every grid value
def far_curve(Z_eval, taus: np.ndarray) -> np.ndarray:
    z = np.sort(np.asarray(Z_eval, dtype=float))
    if len(z) == 0:
        return np.full(len(taus), np.nan)
    k = np.searchsorted(z, taus, side="left")
    return 1.0 - k / len(z)


# Detection rate curve: mean within each edit, then across edits
def tpr_curve(Z_pos, pos_variant, taus: np.ndarray, names: list[str] | None = None):
    names = list(names or variant_names())
    pv = np.asarray(pos_variant)
    Zp = np.asarray(Z_pos, dtype=float)
    per, missing = [], 0
    for v in names:
        z = Zp[pv == v]
        if len(z) == 0:
            missing += 1
            continue
        per.append(far_curve(z, taus))
    if missing or not per:
        return np.full(len(taus), np.nan), missing
    return np.mean(per, axis=0), 0


# Balanced accuracy (TPR + (1 - FAR)) / 2
def ba_curve(far: np.ndarray, tpr: np.ndarray) -> np.ndarray:
    return (np.asarray(tpr, dtype=float) + 1.0 - np.asarray(far, dtype=float)) / 2.0


# Share of tasks whose false alarm rate exceeds the nominal level
def violation_rate(far_rows: np.ndarray, alpha: float) -> np.ndarray:
    return (np.asarray(far_rows, dtype=float) > float(alpha) + 1e-12).mean(axis=0)


# Whether the condition requirement is met
def meets(V, BA, BA_base) -> np.ndarray | bool:
    V = np.asarray(V, dtype=float)
    BA = np.asarray(BA, dtype=float)
    return (V <= DELTA + 1e-12) & (BA >= np.asarray(BA_base, dtype=float) - 1e-12)


# True when the detection rate is zero
def is_no_detection(tpr_mean) -> np.ndarray | bool:
    return np.asarray(tpr_mean, dtype=float) <= 0.0


# Criterion settings recorded in the outputs
def rule_dict() -> dict:
    return {"delta": DELTA,
            "delta_source": "Tong, Feng, Li 2018, Algorithm 1 기본값",
            "performance_baseline": BASELINE_METHOD,
            "performance_baseline_source": "DBpHash 원문 식 (28)을 같은 조건에서 다시 잰 값",
            "summary_metric": "balanced accuracy",
            "summary_metric_source": "DBpHash 원문 Table 3",
            "variants": "transforms.SPEC_V3 전부(23종). 조건별 선정 없음"}
