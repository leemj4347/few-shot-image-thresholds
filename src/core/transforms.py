# Secondary edit set (SECONDARY)
from __future__ import annotations
import os
import numpy as np
import cv2


SPEC_V2 = [
    ("gauss_s05",  "gaussian", {"sigma": 0.05}),
    ("gauss_s10",  "gaussian", {"sigma": 0.10}),
    ("gauss_s20",  "gaussian", {"sigma": 0.20}),
    ("jpeg_q05",   "jpeg",     {"q": 5}),
    ("jpeg_q02",   "jpeg",     {"q": 2}),
    ("median_k05", "median",   {"k": 5}),
    ("median_k09", "median",   {"k": 9}),
    ("scale_f075", "scale",    {"f": 0.75}),
    ("scale_f050", "scale",    {"f": 0.50}),
    ("scale_f025", "scale",    {"f": 0.25}),
    ("rotate_d01", "rotate",   {"deg": 1.0}),
    ("rotate_d03", "rotate",   {"deg": 3.0}),
    ("rotate_d10", "rotate",   {"deg": 10.0}),
    ("crop_f090",  "crop",     {"f": 0.90}),
    ("crop_f080",  "crop",     {"f": 0.80}),
    ("gamma_g080", "gamma",    {"g": 0.80}),
    ("gamma_g125", "gamma",    {"g": 1.25}),
]


SPEC_V3_NEW = [
    ("resize_f075",   "resize",   {"f": 0.75}),
    ("resize_f050",   "resize",   {"f": 0.50}),
    ("cropkeep_f098", "cropkeep", {"f": 0.98}),
    ("cropkeep_f095", "cropkeep", {"f": 0.95}),
    ("cropkeep_f090", "cropkeep", {"f": 0.90}),
    ("shift_f001",    "shift",    {"f": 0.01}),
    ("shift_f003",    "shift",    {"f": 0.03}),
    ("padedge_f002",  "padedge",  {"f": 0.02}),
    ("padedge_f005",  "padedge",  {"f": 0.05}),
    ("rotkeep_d02",   "rotkeep",  {"deg": 2.0}),
    ("rotkeep_d05",   "rotkeep",  {"deg": 5.0}),
    ("text_s10",      "text",     {"scale": 1.0}),
]

SPEC_V3_KEPT = [s for s in SPEC_V2 if s[0] in
                ("gauss_s05", "gauss_s10", "gauss_s20", "jpeg_q05", "jpeg_q02",
                 "median_k05", "median_k09", "gamma_g080", "gamma_g125", "rotate_d01", "rotate_d03")]
SPEC_V3 = SPEC_V3_KEPT + SPEC_V3_NEW


SPEC_CHOICE = os.environ.get("ATTACKS2_SPEC", "v3").lower()
SPEC = {"v2": SPEC_V2, "v3": SPEC_V3}[SPEC_CHOICE]
_ALL = SPEC_V2 + SPEC_V3_NEW
NAMES = [s[0] for s in SPEC]
BY_NAME = {s[0]: s for s in _ALL}


PHOTOMETRIC2 = [n for n, f, _ in _ALL if f in ("gaussian", "jpeg", "median", "gamma", "text")]
GEOMETRIC2 = [n for n, f, _ in _ALL if f in ("scale", "rotate", "crop", "resize", "cropkeep",
                                             "shift", "padedge", "rotkeep")]

REFRAMING = [n for n, f, _ in _ALL if f in ("scale", "crop", "rotate", "padedge")]
CONTENT_PRESERVING = [n for n, f, _ in _ALL if f in ("resize", "cropkeep", "shift", "rotkeep", "text")]

SPEC_VERSIONS = {"v2": "transforms v (17종, 저자 배포본 규약)",
                 "v3": "transforms v (23종, 내용 보존 편집 중심)"}
SPEC_VERSION = SPEC_VERSIONS[SPEC_CHOICE]


# Edited image for one edit name
def apply2(img_bgr: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    if name not in BY_NAME:
        raise ValueError(f"모르는 변형: {name}")
    _, fam, p = BY_NAME[name]
    h, w = img_bgr.shape[:2]

    if fam == "gaussian":
        x = img_bgr.astype(np.float32) / 255.0
        x = np.clip(x + rng.normal(0, p["sigma"], x.shape).astype(np.float32), 0, 1)
        return (x * 255.0 + 0.5).astype(np.uint8)

    if fam == "jpeg":
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(p["q"])])
        if not ok:
            raise RuntimeError("jpeg 인코딩 실패")
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if fam == "median":
        return cv2.medianBlur(img_bgr, int(p["k"]))

    if fam == "scale":
        f = float(p["f"])
        small = cv2.resize(img_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros_like(img_bgr)
        sh, sw = small.shape[:2]
        canvas[:sh, :sw] = small
        return canvas

    if fam == "rotate":
        M = cv2.getRotationMatrix2D((w // 2, h // 2), float(p["deg"]), 1.0)
        return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_CONSTANT)

    if fam == "crop":
        f = float(p["f"])
        ch, cw = max(1, int(h * f)), max(1, int(w * f))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        cut = img_bgr[y0:y0 + ch, x0:x0 + cw]
        canvas = np.zeros_like(img_bgr)
        canvas[:ch, :cw] = cut
        return canvas

    if fam == "resize":
        f = float(p["f"])
        return cv2.resize(img_bgr, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_LINEAR)

    if fam == "cropkeep":
        f = float(p["f"])
        ch, cw = max(1, int(h * f)), max(1, int(w * f))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return img_bgr[y0:y0 + ch, x0:x0 + cw].copy()

    if fam == "shift":
        f = float(p["f"])
        M = np.float32([[1, 0, int(round(w * f))], [0, 1, int(round(h * f))]])
        return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if fam == "padedge":
        f = float(p["f"])
        py, px = int(round(h * f)), int(round(w * f))
        return cv2.copyMakeBorder(img_bgr, py, py, px, px, cv2.BORDER_REPLICATE)

    if fam == "rotkeep":
        deg = float(p["deg"])
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
        r = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a = np.deg2rad(abs(deg))
        c, si = np.cos(a), np.sin(a)
        den = c * c - si * si
        if den <= 1e-6: return r
        nw, nh = int((w * c - h * si) / den), int((h * c - w * si) / den)
        nw, nh = max(8, min(w, nw)), max(8, min(h, nh))
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        return r[y0:y0 + nh, x0:x0 + nw].copy()

    if fam == "text":
        out = img_bgr.copy()
        sc = max(0.4, (w * float(p["scale"])) / 60.0)
        cv2.putText(out, "SAMPLE", (int(w * 0.05), int(h * 0.55)), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, (255, 255, 255), max(1, int(sc * 2)), cv2.LINE_AA)
        return out

    if fam == "gamma":
        g = float(p["g"])
        x = (img_bgr.astype(np.float32) / 255.0) ** g
        return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)

    raise ValueError(fam)


# (idx, root, rel, seed) -> hash rows of the SECONDARY edits
def variants2_one(args):
    import hashcache as HC
    import dbphash as D
    idx, root, rel, seed = args
    img = HC.load_bgr(os.path.join(root, rel))
    if img is None:
        return []
    rows = []
    for name in NAMES:
        rng = np.random.default_rng(seed)
        v = apply2(img, name, rng)
        hsh = D.dbphash(v, HC.CFG["bands"], HC.CFG["interp"])
        rows.append({"file": rel.replace("\\", "/"), "variant": name,
                     "mid_u64": HC.bits_to_u64(hsh["mid"]), "high_u64": HC.bits_to_u64(hsh["high"]),
                     "phash_u64": HC.bits_to_u64(HC.phash_bits(v))})
    return rows
