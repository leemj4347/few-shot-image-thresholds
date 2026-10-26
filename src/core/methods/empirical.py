# Non-learned threshold rules that use only the support set
from __future__ import annotations
import numpy as np
from scipy.stats import binom
import similarity as SIM

DELTA = 0.05


# Largest k with P(Bin(n, eps) <= k) <= delta
def np_umbrella_k(n: int, eps: float, delta: float = DELTA) -> int:
    if n <= 0 or eps <= 0: return -1
    if binom.cdf(0, n, eps) > delta: return -1
    k = 0
    while k + 1 <= n and binom.cdf(k + 1, n, eps) <= delta:
        k += 1
    return k


# Smallest sample size that allows k = 0: ceil(ln(delta) / ln(1 - eps))
def n_min_np(eps: float, delta: float = DELTA) -> int:
    return int(np.ceil(np.log(delta) / np.log(1 - eps)))


# Grid value just above the (k+1)-th largest score
def _tau_from_order(sorted_desc: np.ndarray, k: int, w) -> float:
    if k + 1 > len(sorted_desc): return float("inf")
    v = float(sorted_desc[k])
    return SIM.strict_successor(v,w)


# NP umbrella on query maxima
def tau_C2max(Z_sup: np.ndarray, n: int, alpha: float, w, delta: float = DELTA) -> float:
    z = np.sort(np.asarray(Z_sup, dtype=np.float64)[:n])[::-1]
    k = np_umbrella_k(len(z), alpha, delta)
    if k < 0: return float("inf")
    return _tau_from_order(z, k, w)


# Empirical quantile: smallest grid value with support false-alarm share <= alpha (no violation control)
def tau_C2emp(Z_sup: np.ndarray, n: int, alpha: float, w) -> float:
    return SIM.tau_empirical(np.asarray(Z_sup, dtype=np.float64)[:n], alpha, w)


# Order statistic of pair scores
def tau_C2pair(hist_sup: np.ndarray, M: int, alpha: float, w, delta: float = DELTA):
    eps = 1.0 - (1.0 - alpha) ** (1.0 / M)
    npairs = int(hist_sup.sum())
    need = n_min_np(eps, delta)
    k = np_umbrella_k(npairs, eps, delta)
    if k < 0: return float("inf"), False, need
    d = SIM.GridDist(hist_sup, w=w)

    target = (k + 1) / npairs
    v = d.v
    sf = 1.0 - np.concatenate([[0.0], d.cdf[:-1]])
    idx = np.flatnonzero(sf >= target - 1e-15)
    val = float(v[idx[-1]]) if len(idx) else float(v[0])
    return SIM.strict_successor(val,w), True, need


from scipy.stats import hypergeom


# Largest k with violation probability <= delta for N candidates, n support, target alpha
def np_finite_k(N: int, n: int, alpha: float, delta: float = DELTA) -> int:
    n_eval = N - n
    if n <= 0 or n_eval <= 0 or alpha <= 0: return -1
    m = int(np.floor(alpha * n_eval + 1e-12))

    def v(k: int) -> float:
        draws = k + m + 1
        if draws >= N: return 0.0
        return float(hypergeom.cdf(k, N, n, draws))

    if v(0) > delta: return -1
    k = 0
    while k + 1 <= n - 1 and v(k + 1) <= delta:
        k += 1
    return k


# Smallest support size that allows k = 0 with N candidates
def n_min_finite(N: int, alpha: float, delta: float = DELTA) -> int:
    for n in range(1, N):
        if np_finite_k(N, n, alpha, delta) >= 0: return n
    return -1


# tau from the rank rule adjusted for sampling without replacement
def tau_C2max_fp(Z_sup: np.ndarray, n: int, alpha: float, w, N_cand: int, delta: float = DELTA) -> float:
    z = np.sort(np.asarray(Z_sup, dtype=np.float64)[:n])[::-1]
    k = np_finite_k(int(N_cand), len(z), alpha, delta)
    if k < 0: return float("inf")
    return _tau_from_order(z, k, w)
