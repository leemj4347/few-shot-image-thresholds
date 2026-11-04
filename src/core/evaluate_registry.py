# Registry-level FAR, TPR and balanced accuracy
from __future__ import annotations
import numpy as np
import similarity as SIM

ATTACKS = ["gaussian", "jpeg", "median", "rotate", "scale"]
PHOTO = ["gaussian", "jpeg", "median"]


def score(tau: float, d: dict, alpha: float) -> dict:
    Z = d["Z_eval"]
    Zp = d["Z_pos"]
    var = d["pos_variant"]
    sentinel = not np.isfinite(tau)
    far = 0.0 if sentinel else SIM.far_empirical(Z, tau)
    out = {"tau": tau, "sentinel": bool(sentinel), "FAR": far, "violation": bool(far > alpha + 1e-12),
           "excess": far - alpha, "n_eval": int(len(Z))}
    present = [str(x) for x in np.unique(var)] if len(var) else []
    ph = []
    for a in sorted(set(ATTACKS) | set(present)):
        z = Zp[var == a] if len(Zp) else np.zeros(0)
        v = (0.0 if sentinel else SIM.far_empirical(z, tau)) if len(z) else np.nan
        out[f"TPR_{a}"] = v
        if a in PHOTO and not np.isnan(v): ph.append(v)
    out["TPR_photo3"] = float(np.mean(ph)) if ph else np.nan
    out["n_pos_orig"] = int(d.get("n_pos_orig", 0))
    return out
