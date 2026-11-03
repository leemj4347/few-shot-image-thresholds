# Fixed and development-fitted threshold rules
from __future__ import annotations
import numpy as np
import similarity as SIM

GAMMA_REF = 0.63075
HASH_CONVENTION_BITS = 10
PHOTO = ("gaussian", "jpeg", "median")


# F0 from the support pair counts of the development bundles
def dev_pair_hist(bundles, equal_weight_by_domain: bool = True) -> SIM.GridDist:
    by_dom = {}
    for b in bundles:
        h = b.reg["base"]["hist_sup"]
        if not h: continue
        H = h[max(h)]
        if H.sum() == 0: continue
        by_dom.setdefault(b.domain, []).append(H / H.sum())
    if not by_dom: raise ValueError("no development task has a non-empty pair histogram")
    if equal_weight_by_domain:
        Hn = np.mean(np.stack([np.mean(np.stack(v), axis=0) for v in by_dom.values()]), axis=0)
    else:
        Hn = np.mean(np.stack([x for v in by_dom.values() for x in v]), axis=0)
    return SIM.GridDist(Hn * 1e9, w=bundles[0].w)


# Development bundles -> (impostor pair GridDist, genuine pair scores)
def dev_pair_scores(bundles):
    d0 = dev_pair_hist(bundles)
    gen = np.concatenate([b.S_gen[np.isin(b.gen_variant, PHOTO)] for b in bundles if len(b.S_gen)]) \
        if any(len(b.S_gen) for b in bundles) else np.zeros(0)
    return d0, gen


# Development bundles -> (normal query Z, photometric positive Z)
def dev_query_scores(bundles):
    neg = np.concatenate([b.reg["base"]["Z_eval"] for b in bundles]) if bundles else np.zeros(0)
    pos = np.concatenate([b.reg["base"]["Z_pos"][np.isin(b.reg["base"]["pos_variant"], PHOTO)]
                          for b in bundles if len(b.reg["base"]["Z_pos"])]) if bundles else np.zeros(0)
    return neg, pos


def _ba_argmax(neg, pos, grid):
    if len(neg) == 0 or len(pos) == 0: return float("nan")
    ns = np.sort(neg)
    ps = np.sort(pos)
    far = 1.0 - np.searchsorted(ns, grid - SIM.TOL, side="left") / len(ns)
    tpr = 1.0 - np.searchsorted(ps, grid - SIM.TOL, side="left") / len(ps)
    return float(grid[int(np.argmax(0.5 * (tpr + 1.0 - far)))])


# Query-level BA maximum over all development data
def fit_C0g(dev_bundles) -> float:
    neg, pos = dev_query_scores(dev_bundles)
    return _ba_argmax(neg, pos, SIM.grid_values(dev_bundles[0].w))


# Per-domain refit
def fit_C1(dev_bundles, domain: str) -> float:
    sub = [b for b in dev_bundles if b.domain == domain]
    return fit_C0g(sub) if sub else float("nan")


def _pair_curves(d0: SIM.GridDist, gen: np.ndarray, grid: np.ndarray):
    fpr = np.array([d0.sf(t) for t in grid])
    gs = np.sort(gen)
    tpr = 1.0 - np.searchsorted(gs, grid - SIM.TOL, side="left") / max(len(gs), 1)
    return fpr, tpr


def fit_C0eer(d0, gen, grid) -> float:
    fpr, tpr = _pair_curves(d0, gen, grid)
    return float(grid[int(np.argmin(np.abs(fpr - (1.0 - tpr))))])


def fit_C0j(d0, gen, grid) -> float:
    fpr, tpr = _pair_curves(d0, gen, grid)
    return float(grid[int(np.argmax(tpr - fpr))])


# Otsu threshold on the pooled genuine/impostor score histogram
def fit_C0otsu(d0: SIM.GridDist, gen: np.ndarray, grid) -> float:
    v = d0.v
    w0 = d0.c / max(d0.c.sum(), 1)
    gcnt = np.zeros_like(v)
    if len(gen):
        idx = np.searchsorted(v, np.sort(gen) - SIM.TOL, side="left")
        idx = np.clip(idx, 0, len(v) - 1)
        np.add.at(gcnt, idx, 1)
        gcnt = gcnt / gcnt.sum()
    p = 0.5 * w0 + 0.5 * gcnt if len(gen) else w0
    p = p / p.sum()
    cw = np.cumsum(p)
    cm = np.cumsum(p * v)
    mu = cm[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu * cw - cm) ** 2 / (cw * (1 - cw))
    sb[~np.isfinite(sb)] = -1.0
    k = int(np.argmax(sb))
    i = min(k + 1, len(v) - 1)
    return float(v[i])


# Hash convention: Hamming distance <= 10 bits
def fit_C0h(w=None) -> float:
    return SIM.grid_ceil(1.0 - HASH_CONVENTION_BITS / 64.0, w or SIM.W_FIXED)


# Global scale correction
def tau_C0s(d0: SIM.GridDist, M: int, alpha: float) -> float:
    return SIM.tau_iid(d0, M, alpha)
