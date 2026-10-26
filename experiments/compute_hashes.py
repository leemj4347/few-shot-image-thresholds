# Recompute image hashes using deterministic edits
from __future__ import annotations
import json, os, shutil, datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import experiment as C

POOLS = ("solana", "nft", "disc21_bg")
SUFFIX = ("", "_var", "_var3")
DESIGN_FILES = ("role_table.csv", "nft_meta.csv", "solana_meta.csv", "disc21_bg_list.txt", "disc21_val_list.txt",
                "disc21_split_note.txt")
HASH_COLS = ("mid_u64", "high_u64", "phash_u64")
CHUNK = 100

_W = {}


def _init():
    import cv2
    cv2.setNumThreads(1)
    D, HC, TR = C.runtime_modules()
    _W.update(D=D, HC=HC, TR=TR, v3=set(TR.BY_NAME))


# job = (root, [(file, [(row_index, variant), ...]), ...]) -> list of (row, mid, high, phash, gid, w, h)
def work(job):
    root, files = job
    D, HC, TR = _W["D"], _W["HC"], _W["TR"]
    out = []
    for f, rows in files:
        img = HC.load_bgr(str(Path(root) / f))
        if img is None: raise RuntimeError(f"cannot read {Path(root) / f}")
        seed = HC.variant_seed(f)
        names = [v for _, v in rows]
        prim = {}
        if any(v in D.ATTACKS for v in names):
            rng = np.random.default_rng(seed)
            for a in D.ATTACKS:
                prim[a] = D.attack(img, a, rng, HC.CFG["scale_mode"], HC.CFG["rot_mode"])
        gid = HC.pixel_group_id(img)
        w, h = int(img.shape[1]), int(img.shape[0])
        for r, v in rows:
            if v == "orig": im = img
            elif v in prim: im = prim[v]
            elif v in _W["v3"]: im = TR.apply2(img, v, np.random.default_rng(seed))
            else: raise ValueError(f"unknown variant {v}")
            hs = D.dbphash(im, HC.CFG["bands"], HC.CFG["interp"])
            out.append((int(r), HC.bits_to_u64(hs["mid"]), HC.bits_to_u64(hs["high"]), HC.bits_to_u64(HC.phash_bits(im)),
                        gid if v == "orig" else "", w if v == "orig" else -1, h if v == "orig" else -1))
    return out


# uint64 -> float64 -> integer, as a lossy float conversion would store it (None if it leaves uint64 range)
def float_round_trip(x: int):
    v = int(float(int(x)))
    return v if 0 <= v < 2 ** 64 else None


def rehash_one(name, root, dest, parts, p, ex, smoke):
    old = pd.read_parquet(C.INPUT_CACHE / f"hashes_{name}.parquet")
    for col in HASH_COLS:
        if old[col].dtype != np.uint64: raise TypeError(f"{name}.{col} is not uint64 in the input cache")
    rows = old.index[old.ok].to_numpy()
    by_file = {}
    for r, f, v in zip(rows, old.file.to_numpy()[rows], old.variant.to_numpy()[rows]):
        by_file.setdefault(f, []).append((int(r), v))
    files = sorted(by_file)
    if smoke: files = files[:20]
    chunks = [files[k:k + CHUNK] for k in range(0, len(files), CHUNK)]
    pdir = C.output_path(parts / name)
    pdir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(len(chunks)) if not (pdir / f"part_{i:05d}.npz").exists()]
    tset = set(todo)
    done_rows = sum(len(by_file[f]) for i in range(len(chunks)) if i not in tset for f in chunks[i])
    p.advance("rows", done_rows, file=name, resumed=done_rows)
    futures = {ex.submit(work, (str(root), [(f, by_file[f]) for f in chunks[i]])): i for i in todo}
    from concurrent.futures import as_completed
    for fu in as_completed(futures):
        i = futures[fu]
        res = fu.result()
        arr = np.array([(r, m, h, ph) for r, m, h, ph, *_ in res], dtype=np.uint64)
        meta = [(r, gid, w, hh) for r, _, _, _, gid, w, hh in res if gid]
        tmp = pdir / f"part_{i:05d}.tmp.npz"
        np.savez(tmp, values=arr, orig_rows=np.array([m[0] for m in meta], dtype=np.int64),
                 orig_gid=np.array([m[1] for m in meta], dtype=object), orig_w=np.array([m[2] for m in meta], dtype=np.int64),
                 orig_h=np.array([m[3] for m in meta], dtype=np.int64))
        os.replace(tmp, pdir / f"part_{i:05d}.npz")
        p.advance("rows", len(res), file=name, chunk=i + 1, of=len(chunks))

    new = old.copy()
    keep = np.zeros(len(old), bool)
    vals = {c: old[c].to_numpy().copy() for c in HASH_COLS}
    for i in range(len(chunks)):
        z = np.load(pdir / f"part_{i:05d}.npz", allow_pickle=True)
        v = z["values"]
        r = v[:, 0].astype(np.int64)
        for j, c in enumerate(HASH_COLS): vals[c][r] = v[:, j + 1]
        keep[r] = True
        orr = z["orig_rows"]
        if len(orr):
            if not (old.group_id.to_numpy()[orr] == z["orig_gid"]).all(): raise RuntimeError(f"{name}: pixel group id changed (different image)")
            if not ((old.w.to_numpy()[orr] == z["orig_w"]).all() and (old.h.to_numpy()[orr] == z["orig_h"]).all()):
                raise RuntimeError(f"{name}: image size changed")
    if smoke:
        new = new[keep].reset_index(drop=True)
        old = old[keep].reset_index(drop=True)
        vals = {c: vals[c][keep] for c in HASH_COLS}
    elif not keep[old.ok.to_numpy()].all(): raise RuntimeError(f"{name}: rows missing")
    for c in HASH_COLS: new[c] = np.asarray(vals[c], dtype=np.uint64)
    for c in HASH_COLS:
        if new[c].dtype != np.uint64: raise TypeError(f"{name}.{c} lost uint64")
    target = C.output_path(dest / f"hashes_{name}.parquet")
    tmp = target.with_suffix(".tmp.parquet")
    new.to_parquet(tmp, index=False)
    os.replace(tmp, target)
    back = pd.read_parquet(target)
    for c in HASH_COLS:
        if back[c].dtype != np.uint64 or not (back[c].to_numpy() == new[c].to_numpy()).all(): raise TypeError(f"{name}.{c} changed on write")
    rep = {"rows": int(len(new)), "files": len(files)}
    for c in HASH_COLS:
        a = old[c].to_numpy(np.uint64)
        b = new[c].to_numpy(np.uint64)
        diff = a != b
        rt = [float_round_trip(x) for x in b[diff]]
        rep[c] = {"identical": int((~diff).sum()), "different": int(diff.sum()),
                  "different_equal_to_float64_round_trip_of_new": int(sum(x is not None and x == int(y) for x, y in zip(rt, a[diff])))}
        for kind, mask in (("orig", old.variant.to_numpy() == "orig"), ("edits", old.variant.to_numpy() != "orig")):
            rep[c]["different_" + kind] = int((diff & mask).sum())
    rep["sha256"] = C.sha256(target)
    return rep


def main(args, out, holder):
    smoke = args.smoke
    dest = C.output_path(C.OUT_ROOT / "data" / "hashes_smoke" if smoke else C.ROOT / "data" / "hashes")
    dest.mkdir(parents=True, exist_ok=True)
    parts = C.output_path(C.OUT_ROOT / "data" / "rehash_parts_smoke" if smoke else C.ROOT / "data" / "rehash_parts")
    names = ["nft", "nft_var3"] if smoke else [p + s for p in POOLS for s in SUFFIX]
    root_of = lambda n: C.ROOTS[n.replace("_var3", "").replace("_var", "")]
    total = 0
    for n in names:
        o = pd.read_parquet(C.INPUT_CACHE / f"hashes_{n}.parquet", columns=["file", "ok"])
        total += int(o.ok.sum()) if not smoke else int(o[o.file.isin(sorted(o.file[o.ok].unique())[:20])].ok.sum())
    p = C.Progress("hashes", "해시 계산(DBpHash·pHash)", {"design": 1, "rows": total, "report": 1}, smoke, args.log)
    holder["progress"] = p
    design = {}
    for f in DESIGN_FILES:
        src = C.INPUT_CACHE / f
        dst = C.output_path(dest / f)
        if not dst.exists(): shutil.copyfile(src, dst)
        if C.sha256(src) != C.sha256(dst): raise RuntimeError(f"design file differs: {f}")
        design[f] = C.sha256(dst)
    incl = C.INPUTS / "selection.json"
    dd = C.output_path(C.ROOT / "data" / "design")
    dd.mkdir(parents=True, exist_ok=True)
    if not (dd / "inclusion_fixed.json").exists(): shutil.copyfile(incl, dd / "inclusion_fixed.json")
    design["inclusion_fixed.json"] = C.sha256(dd / "inclusion_fixed.json")
    p.advance("design", files=len(design))
    workers = int(os.environ.get("WORKERS", "0")) or max(1, int((os.cpu_count() or 2) * 0.7))
    report = {"workers": workers, "design_files_sha256": design, "time": datetime.datetime.now().isoformat()}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        for n in names:
            report[n] = rehash_one(n, root_of(n), dest, parts, p, ex, smoke)
    p.advance("report")
    return {n: {"rows": report[n]["rows"], "mid_different": report[n]["mid_u64"]["different"],
                "mid_different_float_round_trip": report[n]["mid_u64"]["different_equal_to_float64_round_trip_of_new"]} for n in names}


if __name__ == "__main__":
    C.run_step("hashes", "해시 계산(DBpHash·pHash)", main)
