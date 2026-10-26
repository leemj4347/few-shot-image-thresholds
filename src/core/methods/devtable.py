# Development-fitted threshold lookup
from __future__ import annotations
import numpy as np
import similarity as SIM
import criteria as CR
from .fixed import GAMMA_REF


# False alarm rate P(Z >= tau) at every grid value from one sort
def _far_curve(Z_eval: np.ndarray, taus: np.ndarray) -> np.ndarray:
    return CR.far_curve(Z_eval, taus)


# Smallest grid value satisfying the constraint
def _pick(far_rows: np.ndarray, ba_rows: np.ndarray, taus: np.ndarray,
          alpha: float, gamma_idx: int) -> float:
    if far_rows.size == 0: return float("nan")
    V = CR.violation_rate(far_rows, alpha)
    if ba_rows.size == 0 or gamma_idx < 0:
        return float("nan")
    BA = np.nanmean(ba_rows, axis=0)
    ok = CR.meets(V, BA, BA[gamma_idx])
    idx = np.flatnonzero(ok)
    return float(taus[idx[0]]) if len(idx) else float("nan")


# Fixed values and lookup table per target from the development bundles
def fit(dev_bundles: list, alphas: list, w: dict, kinds=("base", "grow_i", "grow_ii")) -> dict:
    taus = np.unique(SIM.grid_values(w))
    gamma = SIM.grid_ceil(GAMMA_REF, w)
    gamma_idx = int(np.searchsorted(taus, gamma - SIM.TOL, side="left"))
    if not (0 <= gamma_idx < len(taus)):
        gamma_idx = -1
    names = CR.variant_names()
    per_far, per_ba, pooled_far, pooled_ba = {}, {}, {}, {}
    n_task, n_ba_task, n_missing = 0, 0, 0
    for b in dev_bundles:
        for kind, d in b.reg.items():
            if kind not in kinds or len(d["Z_eval"]) == 0: continue
            n_task += 1
            far = _far_curve(d["Z_eval"], taus)
            key = (b.domain, b.structure, kind, int(b.M))
            per_far.setdefault(key, []).append(far)
            pooled_far.setdefault(kind, []).append(far)
            tpr, miss = CR.tpr_curve(d.get("Z_pos", []), d.get("pos_variant", []), taus, names)
            if miss:
                n_missing += 1
                continue
            n_ba_task += 1
            ba = CR.ba_curve(far, tpr)
            per_ba.setdefault(key, []).append(ba)
            pooled_ba.setdefault(kind, []).append(ba)
    out = {"constant": {}, "table": {}, "by_kind_M": {},
           "info": {"n_conditions": len(per_far), "grid": int(len(taus)),
                    "gamma": float(gamma), "gamma_in_grid": bool(gamma_idx >= 0),
                    "n_tasks": n_task, "n_tasks_with_ba": n_ba_task,
                    "n_tasks_missing_variants": n_missing,
                    "rule": CR.rule_dict()}}
    for a in alphas:
        all_far = np.array([c for lst in pooled_far.values() for c in lst])
        all_ba = np.array([c for lst in pooled_ba.values() for c in lst])
        out["constant"][float(a)] = _pick(all_far, all_ba, taus, float(a), gamma_idx)
        for key in per_far:
            out["table"][key + (float(a),)] = _pick(np.array(per_far[key]),
                                                    np.array(per_ba.get(key, [])),
                                                    taus, float(a), gamma_idx)

        by_far, by_ba = {}, {}
        for (dom, struct, kind, M), lst in per_far.items():
            by_far.setdefault((struct, kind, M), []).extend(lst)
        for (dom, struct, kind, M), lst in per_ba.items():
            by_ba.setdefault((struct, kind, M), []).extend(lst)
        for key, lst in by_far.items():
            out["by_kind_M"][key + (float(a),)] = _pick(np.array(lst),
                                                        np.array(by_ba.get(key, [])),
                                                        taus, float(a), gamma_idx)
    return out


def tau_constant(fitted: dict, alpha: float) -> float:
    return float(fitted["constant"].get(float(alpha), float("nan")))


# Lookup table query
def tau_table(fitted: dict, domain: str, structure: str, kind: str, M: int, alpha: float) -> tuple[float, str]:
    a = float(alpha)
    v = fitted["table"].get((domain, structure, kind, int(M), a), float("nan"))
    if np.isfinite(v): return v, "condition"
    v = fitted["by_kind_M"].get((structure, kind, int(M), a), float("nan"))
    if np.isfinite(v): return v, "kind_M"
    return tau_constant(fitted, a), "constant"
