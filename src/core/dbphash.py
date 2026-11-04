# Dual-band perceptual image hashing
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import dct

EPS = 1e-8
N = 32


# RGB/BGR uint8 → 32×32 grayscale float in [0,1]
def preprocess(img_bgr: np.ndarray, interp: str = "area") -> np.ndarray:
    flag = cv2.INTER_LINEAR if interp == "linear" else cv2.INTER_AREA
    small = cv2.resize(img_bgr, (N, N), interpolation=flag)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return gray.astype(np.float32) / 255.0


def dct2(x: np.ndarray) -> np.ndarray:
    return dct(dct(x.T, norm="ortho").T, norm="ortho")


BANDS = {


    "author": {"mid": (8, 16, 8, 16), "high": (24, 32, 24, 32)},
    "v1_scaled": {"mid": (8, 16, 8, 16), "high": (16, 24, 16, 24)},
    "aux_high_corner": {"mid": (8, 16, 8, 16), "high": (24, 32, 24, 32)},
}


# Energy normalization (step 4) and mean-threshold binarization (Eq. 21, step 5)
def band_bits(C: np.ndarray, box) -> np.ndarray:
    r0, r1, c0, c1 = box
    B = C[r0:r1, c0:c1].flatten()
    Bhat = B / (np.sqrt(np.sum(B ** 2)) + EPS)
    mu = Bhat.mean()
    return (Bhat >= mu).astype(np.uint8)


# Eqs. (21)-(22)
def dbphash(img_bgr: np.ndarray, bands: str = "author", interp: str = "area") -> dict:
    C = dct2(preprocess(img_bgr, interp))
    b = BANDS[bands]
    return {"mid": band_bits(C, b["mid"]), "high": band_bits(C, b["high"])}


def band_dist(h1: dict, h2: dict) -> dict:
    return {b: float(np.count_nonzero(h1[b] != h2[b])) / h1[b].size for b in ("mid", "high")}


def band_sim(d: dict) -> dict:
    return {b: 1.0 - d[b] for b in d}


# sigma_b: standard deviation of d_b over genuine training pairs
def band_weights(genuine_d: list[dict], mode: str = "inv_std") -> dict:
    sig = {b: float(np.std([d[b] for d in genuine_d])) + EPS for b in ("mid", "high")}
    raw = {b: (1.0 / sig[b] if mode == "inv_std" else 1.0 / sig[b] ** 2) for b in sig}
    tot = sum(raw.values())
    return {b: raw[b] / tot for b in raw}, sig


def fused(sim: dict, w: dict) -> float:
    return w["mid"] * sim["mid"] + w["high"] * sim["high"]


def gamma_rule(mu_same: float, mu_diff: float) -> float:
    return 0.5 * (mu_same + mu_diff)


def decide(S: float, gamma: float) -> int:
    return int(S >= gamma)


def attack(img_bgr: np.ndarray, kind: str, rng: np.random.Generator, scale_mode="canvas", rot_mode="reflect") -> np.ndarray:
    if kind == "gaussian":
        x = img_bgr.astype(np.float32) / 255.0
        x = np.clip(x + rng.normal(0, 0.02, x.shape).astype(np.float32), 0, 1)
        return (x * 255.0 + 0.5).astype(np.uint8)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 10])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "median":
        return cv2.medianBlur(img_bgr, 3)
    if kind == "rotate":
        h, w = img_bgr.shape[:2]
        if rot_mode == "author_stored":


            M = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)
            return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
        if rot_mode == "reflect":
            return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if rot_mode == "keep":
            return cv2.warpAffine(img_bgr, M, (w, h), borderValue=(0, 0, 0))
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        return cv2.warpAffine(img_bgr, M, (nw, nh), borderValue=(0, 0, 0))
    if kind == "scale":
        h, w = img_bgr.shape[:2]
        if scale_mode == "author_stored":


            small = cv2.resize(img_bgr, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros_like(img_bgr)
            sh, sw = small.shape[:2]
            canvas[:sh, :sw] = small
            return canvas
        small = cv2.resize(img_bgr, (max(1, int(w * 0.9)), max(1, int(h * 0.9))), interpolation=cv2.INTER_LINEAR)
        if scale_mode == "resize":
            return small
        canvas = np.zeros_like(img_bgr)
        sh, sw = small.shape[:2]
        y0, x0 = (h - sh) // 2, (w - sw) // 2
        canvas[y0:y0 + sh, x0:x0 + sw] = small
        return canvas
    raise ValueError(kind)

ATTACKS = ["gaussian", "jpeg", "median", "rotate", "scale"]
PHOTOMETRIC = ["gaussian", "jpeg", "median"]


def load_bgr(path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
