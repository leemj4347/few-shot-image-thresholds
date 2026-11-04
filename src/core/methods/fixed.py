# Fixed and development-fitted thresholds
from __future__ import annotations
import numpy as np
import similarity as SIM

GAMMA_REF = 0.63075
HASH_CONVENTION_BITS = 10
PHOTO = ("gaussian", "jpeg", "median")


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


def dev_pair_scores(bundles):
    d0 = dev_pair_hist(bundles)
    gen = np.concatenate([b.S_gen[np.isin(b.gen_variant, PHOTO)] for b in bundles if len(b.S_gen)]) \
        if any(len(b.S_gen) for b in bundles) else np.zeros(0)
    return d0, gen


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


def fit_C0h(w=None) -> float:
    return SIM.grid_ceil(1.0 - HASH_CONVENTION_BITS / 64.0, w or SIM.W_FIXED)
