# Hashes and SSCD descriptors for the DISC21 external test
from __future__ import annotations
import json, os, zipfile, datetime
from pathlib import Path
import numpy as np, pandas as pd
import experiment as C
VAL_LIST = C.P02 / "disc21_val_list.txt"
REF_ROOT = C.ROOTS["disc21_bg"]
QZIP = C.DISC21 / "dev_queries.zip"
GT = C.DISC21 / "dev_ground_truth.csv"
BATCH = 128


# index.parquet (name) + mid/high uint64 + SSCD float32 + ok/done masks, resumable, inside this folder
class FeatStore:

    def __init__(self, folder: Path, names=None, dim=512):
        self.folder = C.output_path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        ip = self.folder / "index.parquet"
        if names is not None and (not ip.exists()):
            C.save_parquet(pd.DataFrame({"name": list(names)}), ip)
            n = len(names)
            np.lib.format.open_memmap(self.folder / "emb.npy", mode="w+", dtype=np.float32, shape=(n, dim))
            for f in ("mid", "high"):
                np.lib.format.open_memmap(self.folder / f"{f}.npy", mode="w+", dtype=np.uint64, shape=(n,))
            np.save(self.folder / "done.npy", np.zeros(n, bool))
            np.save(self.folder / "ok.npy", np.zeros(n, bool))
        self.index = pd.read_parquet(ip)
        if names is not None and list(self.index.name) != list(names):
            raise RuntimeError(f"index changed in {self.folder}")
        mode = "r+" if names is not None else "r"
        self.emb = np.load(self.folder / "emb.npy", mmap_mode=mode)
        self.mid = np.load(self.folder / "mid.npy", mmap_mode=mode)
        self.high = np.load(self.folder / "high.npy", mmap_mode=mode)
        self.done = np.load(self.folder / "done.npy")
        self.ok = np.load(self.folder / "ok.npy")

    def save(self):
        for a in (self.emb, self.mid, self.high):
            a.flush()
        for nm, arr in (("done", self.done), ("ok", self.ok)):
            tmp = self.folder / f"{nm}.tmp.npy"
            np.save(tmp, arr)
            tmp.replace(self.folder / f"{nm}.npy")

    @staticmethod
    def open_read(folder: Path):
        s = FeatStore(folder)
        if not s.done.all():
            raise RuntimeError(f"features incomplete in {folder}: {int((~s.done).sum())} missing (run the disc21_prepare step)")
        return s


# read_bytes(i) -> bytes; decode + DBpHash + SSCD preprocessing in threads, SSCD batches on the model device
def run_store(name, st, read_bytes, D, HC, model, p, ex):
    todo = np.flatnonzero(~st.done)
    p.advance(name, int(st.done.sum()), resumed=int(st.done.sum()))

    def prep(buf):
        img = C.load_bgr_bytes(buf) if buf is not None else None
        if img is None:
            return (None, 0, 0)
        h = D.dbphash(img, HC.CFG["bands"], HC.CFG["interp"])
        return (C.sscd_tensor(img), HC.bits_to_u64(h["mid"]), HC.bits_to_u64(h["high"]))
    bad = 0
    chunks = [todo[b0:b0 + BATCH] for b0 in range(0, len(todo), BATCH)]
    submit = lambda ix: [ex.submit(prep, read_bytes(int(i))) for i in ix]
    pending = submit(chunks[0]) if chunks else []
    for k, ix in enumerate(chunks):
        res = [f.result() for f in pending]
        pending = submit(chunks[k + 1]) if k + 1 < len(chunks) else []
        good = [(i, t, m, h) for i, (t, m, h) in zip(ix, res) if t is not None]
        E = C.sscd_embed_tensors(model, [g[1] for g in good]) if good else []
        for (i, _, m, h), e in zip(good, E):
            st.mid[i] = np.uint64(m)
            st.high[i] = np.uint64(h)
            st.emb[i] = e
            st.ok[i] = True
        for i, (t, _, _) in zip(ix, res):
            if t is None:
                bad += 1
                st.ok[i] = False
            st.done[i] = True
        p.advance(name, len(ix), last=str(st.index.name.iat[int(ix[-1])]), unreadable=bad)
        if k % 20 == 0:
            st.save()
    st.save()
    return bad


def main(args, out, holder):
    D, HC, _ = C.runtime_modules()
    info = json.loads((C.WEIGHTS.parent / "weights_info.json").read_text(encoding="utf-8"))
    refs = [Path(s.strip()).stem for s in VAL_LIST.read_text(encoding="utf-8").splitlines() if s.strip()]
    zf = zipfile.ZipFile(QZIP)
    members = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    if len(refs) != 150000 or len(members) != 50000:
        raise ValueError(f"unexpected counts: refs {len(refs)}, queries {len(members)}")
    qnames = sorted(members)
    if args.smoke:
        refs, qnames = (refs[:20], qnames[:20])
    base = C.OUT_ROOT / "data" / "disc21_features_smoke" if args.smoke else C.ROOT / "data" / "disc21_features"
    p = C.Progress(
        "disc21_prepare",
        "DISC21 외부 시험 특징(해시·SSCD)",
        {"references": len(refs), "queries": len(qnames), "check": 1},
        args.smoke,
        args.log,
    )
    holder["progress"] = p
    model = C.sscd_load()
    rs = FeatStore(base / "references", refs, info["dim"])
    qs = FeatStore(base / "queries", qnames, info["dim"])

    def ref_bytes(i):
        f = REF_ROOT / (rs.index.name.iat[i] + ".jpg")
        try:
            return f.read_bytes()
        except OSError:
            return None
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "0")) or max(2, int((os.cpu_count() or 4) * 0.7))) as ex:
        run_store("references", rs, ref_bytes, D, HC, model, p, ex)
        run_store("queries", qs, lambda i: zf.read(members[qs.index.name.iat[i]]), D, HC, model, p, ex)
    zf.close()
    return {"unreadable_references": int((~rs.ok).sum()),
            "unreadable_queries": int((~qs.ok).sum())}

if __name__ == "__main__":
    C.run_step("disc21_prepare", "DISC21 외부 시험 특징(해시·SSCD)", main)
