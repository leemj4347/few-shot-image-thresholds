# Baseline: MLP on fixed summary statistics with meta calibration Lambda
from __future__ import annotations
import numpy as np
import similarity as SIM
from .mlp import MLP
from . import calibration as FC
HIDDEN = 256
DEPTH = 2
EPOCHS = 400
FEATURES = ["os_k", "os_2k", "os_4k", "zmax", "median", "std", "skew", "log10M", "alpha", "log10n"]


def _skew(z: np.ndarray) -> float:
    s = z.std()
    return float(np.mean(((z - z.mean()) / s) ** 3)) if s > 1e-12 else 0.0


# Fixed summary statistics T
def features(Z_sup: np.ndarray, n: int, M: int, alpha: float) -> np.ndarray:
    z = np.sort(np.asarray(Z_sup, dtype=np.float64)[:n])[::-1]
    if len(z) == 0:
        z = np.zeros(1)
    m = len(z)
    k = max(1, m // 8)

    def top(i: int) -> float:
        return float(z[min(i, m) - 1])
    return np.array([
        top(k),
        top(2 * k),
        top(4 * k),
        float(z[0]),
        float(np.median(z)),
        float(z.std()),
        _skew(z),
        np.log10(M),
        alpha,
        np.log10(max(n, 1)),
    ])


# Empirical quantile q_beta from the extra normal scores of meta-training tasks
def build_training_set(train_bundles, alphas, n_list, kind: str="base"):
    X, y, meta = ([], [], [])
    for b in train_bundles:
        d = b.reg.get(kind)
        if d is None or len(d["Z_eval"]) == 0:
            continue
        for a in alphas:
            t = FC.empirical_quantile(d["Z_eval"], FC.beta_fraction(a))
            if not np.isfinite(t):
                continue
            for n in n_list:
                if n > len(d["Z_sup"]):
                    continue
                X.append(features(d["Z_sup"], n, d["M"], a))
                y.append(t)
                meta.append({"domain": b.domain, "M": d["M"], "seed": b.seed, "alpha": a, "n": n})
    return (np.array(X), np.array(y), meta)


# Train on meta-training tasks
def fit(train_bundles, alphas, n_list, seed: int=0, hidden: int=HIDDEN, epochs: int=EPOCHS, kind: str="base", progress=None):
    X, y, meta = build_training_set(train_bundles, alphas, n_list, kind)
    if len(y) < 20:
        return (None, {"n_train": len(y)})
    m = MLP(X.shape[1], hidden=hidden, seed=seed, depth=DEPTH).fit(X, y, epochs=epochs, seed=seed, progress=progress)
    pred = m.predict_raw(X)
    return (
        m,
        {
            "n_train": int(len(y)),
            "hidden": hidden,
            "depth": DEPTH,
            "epochs": epochs,
            "features": FEATURES,
            "train_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
        },
    )


# Prediction before grid correction
def predict_raw(model, Z_sup, n: int, M: int, alpha: float) -> float:
    if model is None:
        return float("nan")
    return float(model.predict_raw(features(Z_sup, n, M, alpha)[None, :])[0])


# Fisch Eq (12) right ECDF: P(Z <= v), no tolerance
def _cdf_at(Z_eval: np.ndarray, v: float) -> float:
    return FC.cdf_at(Z_eval, v)


# Eq (12) weighted breakpoint minimum; lo/hi/iters do not bound Lambda
def calibrate(model, cal_bundles, alpha: float, n: int, kind: str="base", lo: float=-1.0, hi: float=1.0, iters: int=48) -> dict:
    rows = []
    for b in cal_bundles:
        d = b.reg.get(kind)
        if d is None or len(d["Z_eval"]) == 0 or n > len(d["Z_sup"]):
            continue
        q = predict_raw(model, d["Z_sup"], n, d["M"], alpha)
        if not np.isfinite(q):
            raise ValueError("nonfinite model prediction in calibration")
        rows.append((q, d["Z_eval"]))
    return FC.calibrate_predictions(rows, alpha)


# Map the corrected boundary to the next score-grid value
def tau(model, lam: float, Z_sup, n: int, M: int, alpha: float, w) -> float:
    if model is None:
        raise ValueError("missing model")
    q = predict_raw(model, Z_sup, n, M, alpha)
    return SIM.strict_successor(FC.corrected_boundary(q, lam), w)


def fit_C3(train_bundles, alphas, n_list, seed: int=0, hidden: int=HIDDEN, epochs: int=EPOCHS):
    return fit(train_bundles, alphas, n_list, seed=seed, hidden=hidden, epochs=epochs)


# Uncalibrated variant (diagnostic)
def tau_C3(model, Z_sup, n: int, M: int, alpha: float, w) -> float:
    q = predict_raw(model, Z_sup, n, M, alpha)
    if not np.isfinite(q):
        return float("inf")
    return SIM.strict_successor(q, w)
