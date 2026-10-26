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


# Per-method summary: violation rate V, mean excess on violating tasks, photometric TPR on compliant tasks, sentinel share
def summarize(df, by=("method", "kind", "alpha", "n")):
    import pandas as pd
    g = df.groupby(list(by))
    rows = []
    for k, s in g:
        ok = s[~s.violation]
        rows.append(dict(zip(by, k if isinstance(k, tuple) else (k,)),
                         n_tasks=len(s), V=float(s.violation.mean()),
                         excess_mean_violating=float(s.loc[s.violation, "excess"].mean()) if s.violation.any() else 0.0,
                         excess_mean_all=float(s.excess.mean()),
                         TPR_photo3_ok=float(ok.TPR_photo3.mean()) if len(ok) else np.nan,
                         TPR_photo3_all=float(s.TPR_photo3.mean()),
                         sentinel_rate=float(s.sentinel.mean()),
                         FAR_mean=float(s.FAR.mean())))
    return pd.DataFrame(rows)
