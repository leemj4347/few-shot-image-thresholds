# Summary statistics for the H256 predictor
from __future__ import annotations
import numpy as np

FEATURES = ["os_k", "os_2k", "os_4k", "zmax", "median", "std", "skew", "log10M", "alpha", "log10n"]


def _skew(z: np.ndarray) -> float:
    s = z.std()
    return float(np.mean(((z - z.mean()) / s) ** 3)) if s > 1e-12 else 0.0


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
