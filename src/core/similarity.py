# Weighted Hamming similarity and registry score grids
from __future__ import annotations
import numpy as np
W_FIXED = {"mid": 0.49900097706058244, "high": 0.5009990229394177}
W_SINGLE = {"mid": 1.0, "high": 0.0}
SCHEME_W = {"dbp": W_FIXED, "phash": W_SINGLE, "iphash": W_SINGLE}
NB = 64
TOL = 0.0
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# uint64 array -> bit count (uint8)
def popcount_u64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.uint8)
    b = np.ascontiguousarray(x).view(np.uint8).reshape(x.shape + (8,))
    return _LUT[b].sum(axis=-1).astype(np.uint8)


# 65x65 grid G[dm, dh] = S
def score_grid(w: dict=W_FIXED) -> np.ndarray:
    d = np.arange(NB + 1, dtype=np.float64)
    return (w["mid"] * (1.0 - d[:, None] / NB) + w["high"] * (1.0 - d[None, :] / NB)).astype(np.float64)


# Sorted unique values of S
def grid_values(w: dict=W_FIXED) -> np.ndarray:
    return np.unique(score_grid(w))


# Smallest grid value >= tau
def grid_ceil(tau: float, w: dict=W_FIXED) -> float:
    g = grid_values(w)
    i = np.searchsorted(g, tau, side="left")
    return float(g[i]) if i < len(g) else float("inf")


# (nQ,) × (nR,) → d_mid, d_high (nQ, nR) uint8
def dist_pairs(qm: np.ndarray, qh: np.ndarray, rm: np.ndarray, rh: np.ndarray):
    dm = popcount_u64(qm[:, None] ^ rm[None, :])
    dh = popcount_u64(qh[:, None] ^ rh[None, :])
    return (dm, dh)


# Same float64 operation order as score_grid so values match the grid
def sim_from_dist(dm: np.ndarray, dh: np.ndarray, w: dict=W_FIXED) -> np.ndarray:
    return w["mid"] * (1.0 - dm.astype(np.float64) / NB) + w["high"] * (1.0 - dh.astype(np.float64) / NB)


def sim_matrix(qm, qh, rm, rh, w: dict=W_FIXED) -> np.ndarray:
    dm, dh = dist_pairs(qm, qh, rm, rh)
    return sim_from_dist(dm, dh, w)


# Query chunk size (rows)
def auto_chunk(M: int, budget_elems: int=4000000) -> int:
    return int(max(1, min(256, budget_elems // max(M, 1))))


# Per query Z = max_r S(q, r) and argmax
def query_max(qm, qh, rm, rh, w: dict=W_FIXED, chunk: int | None=None, topk: int=0, hist: bool=False):
    nQ = len(qm)
    if chunk is None:
        chunk = auto_chunk(len(rm))
    Z = np.empty(nQ, dtype=np.float64)
    A = np.empty(nQ, dtype=np.int64)
    TK_i = np.empty((nQ, topk), dtype=np.int64) if topk else None
    TK_s = np.empty((nQ, topk), dtype=np.float64) if topk else None
    H = np.zeros((NB + 1, NB + 1), dtype=np.int64) if hist else None
    for c0 in range(0, nQ, chunk):
        sl = slice(c0, c0 + chunk)
        dm, dh = dist_pairs(qm[sl], qh[sl], rm, rh)
        S = sim_from_dist(dm, dh, w)
        A[sl] = S.argmax(axis=1)
        Z[sl] = S[np.arange(S.shape[0]), A[sl]]
        if topk:
            k = min(topk, S.shape[1])
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
            sc = np.take_along_axis(S, idx, axis=1)
            order = np.argsort(-sc, axis=1)
            TK_i[sl, :k] = np.take_along_axis(idx, order, axis=1)
            TK_s[sl, :k] = np.take_along_axis(sc, order, axis=1)
            if k < topk:
                TK_i[sl, k:] = -1
                TK_s[sl, k:] = np.nan
        if hist:
            H += np.bincount(
                dm.astype(np.int32).ravel() * (NB + 1) + dh.astype(np.int32).ravel(),
                minlength=(NB + 1) * (NB + 1),
            ).reshape(NB + 1, NB + 1)
    out = {"Z": Z, "argmax": A}
    if topk:
        out["topk_idx"] = TK_i
        out["topk_sim"] = TK_s
    if hist:
        out["hist"] = H
    return out


# Pair score distribution F0 as a 65x65 count matrix
class GridDist:

    def __init__(self, H: np.ndarray, w: dict=W_FIXED):
        G = score_grid(w).ravel()
        cnt = H.ravel().astype(np.float64)
        order = np.argsort(G)
        self.vals = G[order]
        self.cnt = cnt[order]
        uv, inv = np.unique(self.vals, return_inverse=True)
        c = np.zeros(len(uv))
        np.add.at(c, inv, self.cnt)
        self.v = uv
        self.c = c
        self.n = c.sum()
        self.cdf = np.cumsum(c) / max(self.n, 1)

    # F₀(τ−) = P(S < τ)
    def cdf_left(self, tau: float) -> float:
        i = np.searchsorted(self.v, tau, side="left")
        return float(self.cdf[i - 1]) if i > 0 else 0.0

    # P(S ≥ τ)
    def sf(self, tau: float) -> float:
        return 1.0 - self.cdf_left(tau)

    # Smallest grid value v with P(S < v) >= p (left-limit on the grid)
    def quantile_ge(self, p: float) -> float:
        if p <= 0:
            return float(self.v[0])
        i = np.searchsorted(self.cdf, p, side="left")
        return float(self.v[min(i + 1, len(self.v) - 1)]) if i < len(self.v) - 1 else float("inf")


# Query false alarm rate under independence: 1 - F0(tau-)^M
def far_iid(F0_left: float, M: int) -> float:
    return 1.0 - F0_left ** M


# Smallest grid value with 1 - F0(tau-)^M <= alpha
def tau_iid(dist: GridDist, M: int, alpha: float) -> float:
    need = (1.0 - alpha) ** (1.0 / M)
    return dist.quantile_ge(need)


def far_empirical(Z: np.ndarray, tau: float) -> float:
    return float(np.mean(Z >= tau)) if len(Z) else float("nan")


# Smallest grid value with FAR(tau) <= alpha on normal queries Z
def tau_empirical(Z: np.ndarray, alpha: float, w: dict=W_FIXED) -> float:
    from fractions import Fraction
    a = Fraction(str(alpha))
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    zs = np.sort(np.asarray(Z, dtype=np.float64))
    n = len(zs)
    if not n:
        return float("inf")
    if not np.isfinite(zs).all():
        raise ValueError("finite scores required")
    k = a.numerator * n // a.denominator
    cut = float(zs[n - 1 - k])
    v = strict_successor(cut, w)
    if int(np.count_nonzero(zs >= v)) > k:
        raise RuntimeError("Evaluation threshold exceeds the allowed false-alarm count")
    return v


# Smallest raw grid value strictly greater than u, or +inf
def strict_successor(u: float, w: dict=W_FIXED) -> float:
    if np.isnan(u):
        raise ValueError("NaN boundary")
    g = grid_values(w)
    i = np.searchsorted(g, u, side="right")
    return float(g[i]) if i < len(g) else float("inf")
