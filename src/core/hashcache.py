# Image decoding, perceptual hashes and deterministic variants
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbphash as D
import imagehash
from PIL import Image as PILImage

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
CFG = {"bands": "author", "interp": "area", "scale_mode": "author_stored", "rot_mode": "author_stored"}

HASH_SPEC = "dbp(author,area)+phash(imagehash,8)+iphash=high v"
ATTACKS = D.ATTACKS


# Image read that handles non-ASCII paths
def load_bgr(path: str) -> np.ndarray | None:
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if im is None:

        try:
            from PIL import Image
            pim = Image.open(path)
            pim.seek(0)
            if pim.mode in ("RGBA", "LA", "P"):
                pim = pim.convert("RGBA")
                bg = Image.new("RGBA", pim.size, (255, 255, 255, 255))
                bg.alpha_composite(pim)
                pim = bg.convert("RGB")
            else:
                pim = pim.convert("RGB")
            return cv2.cvtColor(np.array(pim), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    if im.dtype != np.uint8:
        im = (im.astype(np.float64) / (np.iinfo(im.dtype).max / 255.0) + 0.5).astype(np.uint8) if np.issubdtype(im.dtype, np.integer) else np.clip(im * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        a = im[:, :, 3:4].astype(np.float32) / 255.0
        rgb = im[:, :, :3].astype(np.float32)
        out = rgb * a + 255.0 * (1.0 - a)
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
    if im.shape[2] == 3:
        return im
    return None


# 64-bit pHash
def phash_bits(img_bgr: np.ndarray) -> np.ndarray:
    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("L")
    return np.array(imagehash.phash(pil, hash_size=8).hash, dtype=np.uint8).flatten()


# 64 uint8 bits -> uint64 (MSB first)
def bits_to_u64(bits: np.ndarray) -> int:
    return int(np.packbits(bits.astype(np.uint8)).view(">u8")[0])


def pixel_group_id(img: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(img).tobytes())
    h.update(str(img.shape).encode())
    return h.hexdigest()[:16]


# (idx, root, rel) -> row dict
def hash_one(args):
    idx, root, rel = args
    p = os.path.join(root, rel)
    img = load_bgr(p)
    if img is None:
        return {
            "file": rel.replace("\\", "/"),
            "ok": False,
            "group_id": "",
            "mid_u64": 0,
            "high_u64": 0,
            "phash_u64": 0,
            "w": 0,
            "h": 0,
        }
    hsh = D.dbphash(img, CFG["bands"], CFG["interp"])
    return {"file": rel.replace("\\", "/"), "ok": True, "group_id": pixel_group_id(img),
            "mid_u64": bits_to_u64(hsh["mid"]), "high_u64": bits_to_u64(hsh["high"]),
            "phash_u64": bits_to_u64(phash_bits(img)),
            "w": int(img.shape[1]), "h": int(img.shape[0])}


# Random seed for the edited variants
def variant_seed(rel: str) -> int:
    return int(hashlib.sha256(rel.replace("\\", "/").encode("utf-8")).hexdigest()[:8], 16)


# (idx, root, rel, seed) -> hash rows of the five DBpHash reference edits
def variants_one(args):
    idx, root, rel, seed = args
    img = load_bgr(os.path.join(root, rel))
    if img is None:
        return []
    rng = np.random.default_rng(seed)
    rows = []
    for a in ATTACKS:
        v = D.attack(img, a, rng, CFG["scale_mode"], CFG["rot_mode"])
        hsh = D.dbphash(v, CFG["bands"], CFG["interp"])
        rows.append({"file": rel.replace("\\", "/"), "variant": a,
                     "mid_u64": bits_to_u64(hsh["mid"]), "high_u64": bits_to_u64(hsh["high"]),
                     "phash_u64": bits_to_u64(phash_bits(v))})
    return rows


def list_files(root: str, list_file: str | None, meta: pd.DataFrame | None, file_col: str | None) -> list[str]:
    if meta is not None:
        rels = [str(x) for x in meta[file_col].tolist()]
    elif list_file:
        rels = [ln.strip() for ln in open(list_file, encoding="utf-8") if ln.strip()]
    else:
        rels = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    rels.append(os.path.relpath(os.path.join(dp, fn), root))
    rels = sorted(set(r.replace("\\", "/") for r in rels))
    return rels


PARTIAL_EXIT = 75


def _parts_setup(out: str, manifest: dict, part_dir_name: str | None = None):
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    part_dir = outp.parent / (part_dir_name or (outp.stem + "_parts"))
    part_dir.mkdir(exist_ok=True)
    mpath = part_dir / "manifest.json"
    if mpath.exists():
        old = json.load(open(mpath, encoding="utf-8"))
        if old != manifest:
            raise RuntimeError(f"part dir {part_dir} was built for a different input ({old}); use another --out")
    else:
        if any(part_dir.glob("part_*.parquet")):
            raise RuntimeError(f"part dir {part_dir} has parts but no manifest; move it aside")
        json.dump(manifest, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return outp, part_dir


# Process one chunk in smaller batches
def _chunk_rows(part_dir, c0, tasks, fn, workers, mapchunk, flatten, t0, max_seconds, sub=50):
    ipp = part_dir / f"inprog_{c0:09d}.parquet"
    ipj = part_dir / f"inprog_{c0:09d}.json"
    rows = []
    i = 0
    if ipp.exists() and ipj.exists():
        i = int(json.load(open(ipj, encoding="utf-8"))["n_tasks"])
        if 0 < i <= len(tasks):
            rows = pd.read_parquet(ipp).to_dict("records")
            pass
        else:
            i = 0
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        while i < len(tasks):
            b = tasks[i:i + sub]

            mc = max(1, min(mapchunk, len(b) // max(1, workers * 2)))
            res = list(ex.map(fn, b, chunksize=mc)) if ex else [fn(t) for t in b]
            rows.extend([r for rs in res for r in rs] if flatten else res)
            i += len(b)
            if i < len(tasks) and max_seconds and time.time() - t0 > max_seconds:
                pd.DataFrame(rows).to_parquet(ipp, index=False)
                json.dump({"n_tasks": i}, open(ipj, "w", encoding="utf-8"))
                return None, i
    finally:
        if ex: ex.shutdown()
    return rows, i


def build_cache(dataset: str, root: str, out: str, list_file=None, meta_csv=None, file_col=None,
                collection_col=None, mint_col=None, workers=1, chunk=5000, max_seconds: float | None = None) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv, dtype=str) if meta_csv else None
    rels = list_files(root, list_file, meta, file_col)
    if not rels: raise ValueError(f"[{dataset}] no image files under {root}")

    manifest = {"dataset": dataset, "root": os.path.abspath(root), "chunk": chunk, "n": len(rels),
                "hash_spec": HASH_SPEC,
                "rels_sha256": hashlib.sha256("\n".join(rels).encode("utf-8")).hexdigest()}
    outp, part_dir = _parts_setup(out, manifest)
    parts = []
    t0 = time.time()
    for c0 in range(0, len(rels), chunk):
        pf = part_dir / f"part_{c0:09d}.parquet"
        if pf.exists():
            parts.append(pf)
            continue
        tasks = [(c0 + i, root, r) for i, r in enumerate(rels[c0:c0 + chunk])]
        rows, ndone = _chunk_rows(part_dir, c0, tasks, hash_one, workers, 64, False, t0, max_seconds, sub=25)
        if rows is None:
            sys.exit(PARTIAL_EXIT)
        pd.DataFrame(rows).to_parquet(pf, index=False)
        parts.append(pf)
        done = min(c0 + chunk, len(rels))
        pass
        if max_seconds and time.time() - t0 > max_seconds and done < len(rels):
            sys.exit(PARTIAL_EXIT)
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["dataset"] = dataset
    df["variant"] = "orig"

    df["collection"] = ""
    df["mint_time"] = ""
    if meta is not None:
        m = meta.copy()
        m["file"] = m[file_col].astype(str).str.replace("\\", "/", regex=False)
        keep = {"file": "file"}
        if collection_col: keep[collection_col] = "collection"
        if mint_col: keep[mint_col] = "mint_time"
        m = m[list(keep)].rename(columns=keep).drop_duplicates("file")
        df = df.drop(columns=[c for c in ("collection", "mint_time") if c in m.columns]).merge(m, on="file", how="left")
        df["collection"] = df["collection"].fillna("")
        df["mint_time"] = df.get("mint_time", "").fillna("")

    df = df.sort_values("file").reset_index(drop=True)
    df["is_rep"] = False
    okm = df["ok"].values
    first = df[okm].groupby("group_id")["file"].transform("min") == df.loc[okm, "file"]
    df.loc[df.index[okm], "is_rep"] = first.values
    df = df[["dataset", "file", "variant", "group_id", "is_rep", "collection", "mint_time",
             "mid_u64", "high_u64", "phash_u64", "w", "h", "ok"]]
    for c in ("mid_u64", "high_u64", "phash_u64"): df[c] = df[c].astype("uint64")
    df.to_parquet(out, index=False)
    return df


# Apply the five DBpHash reference edits to cached originals and hash them
def build_variants(cache_parquet: str, root: str, out: str, only_reps=True, limit=None, workers=1,
                   chunk: int = 300, max_seconds: float | None = None) -> pd.DataFrame:
    base = pd.read_parquet(cache_parquet)
    base = base[base.ok & (base.variant == "orig")]
    if only_reps: base = base[base.is_rep]
    base = base.sort_values("file").reset_index(drop=True)
    if limit: base = base.iloc[:limit]
    files = base["file"].tolist()
    manifest = {"kind": "variants", "cache": os.path.basename(cache_parquet), "root": os.path.abspath(root),
                "chunk": chunk, "n": len(files), "limit": limit, "hash_spec": HASH_SPEC,
                "files_sha256": hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()}
    outp, part_dir = _parts_setup(out, manifest)
    parts = []
    t0 = time.time()
    for c0 in range(0, len(files), chunk):
        pf = part_dir / f"part_{c0:09d}.parquet"
        if pf.exists():
            parts.append(pf)
            continue
        tasks = [(c0 + i, root, r, variant_seed(r)) for i, r in enumerate(files[c0:c0 + chunk])]
        rows, ndone = _chunk_rows(part_dir, c0, tasks, variants_one, workers, 8, True, t0, max_seconds, sub=8)
        if rows is None:
            sys.exit(PARTIAL_EXIT)
        pd.DataFrame(rows).to_parquet(pf, index=False)
        parts.append(pf)
        done = min(c0 + chunk, len(files))
        pass
        if max_seconds and time.time() - t0 > max_seconds and done < len(files):
            sys.exit(PARTIAL_EXIT)
    v = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()
    inherit = base[["file", "dataset", "group_id", "is_rep", "collection", "mint_time", "w", "h"]]
    v = v.merge(inherit, on="file", how="left")
    v["ok"] = True
    v = v[["dataset", "file", "variant", "group_id", "is_rep", "collection", "mint_time", "mid_u64", "high_u64", "phash_u64", "w", "h", "ok"]]
    for c in ("mid_u64", "high_u64", "phash_u64"): v[c] = v[c].astype("uint64")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    v.to_parquet(out, index=False)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list")
    ap.add_argument("--meta")
    ap.add_argument("--file-col")
    ap.add_argument("--collection-col")
    ap.add_argument("--mint-col")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--variants", help="Cache parquet path; apply five edits to representative originals")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--all-files", action="store_true", help="Apply edits to all images")
    ap.add_argument("--max-seconds", type=float, help="Save completed work after this time limit and exit with code 75")
    a = ap.parse_args()
    if a.variants:
        build_variants(a.variants, a.root, a.out, only_reps=not a.all_files, limit=a.limit, workers=a.workers,
                       chunk=a.chunk if a.chunk != 5000 else 300, max_seconds=a.max_seconds)
    else:
        assert a.dataset, "--dataset is required"
        build_cache(a.dataset, a.root, a.out, a.list, a.meta, a.file_col, a.collection_col, a.mint_col, a.workers,
                    a.chunk, a.max_seconds)
