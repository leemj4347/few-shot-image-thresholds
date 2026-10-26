# SSCD descriptors for originals and edited copies
from __future__ import annotations
import json, os
import numpy as np, pandas as pd
import experiment as C
POOLS = {
    "solana": ("hashes_solana.parquet", ["hashes_solana_var.parquet", "hashes_solana_var3.parquet"]),
    "nft": ("hashes_nft.parquet", ["hashes_nft_var.parquet", "hashes_nft_var3.parquet"]),
    "disc21_bg": ("hashes_disc21_bg.parquet", ["hashes_disc21_bg_var.parquet", "hashes_disc21_bg_var3.parquet"]),
}


def pool_index(name):
    cache, vfiles = POOLS[name]
    o = pd.read_parquet(C.P02 / cache)
    o = o[o.ok & (o.variant == "orig") & o.is_rep][["dataset", "file", "variant", "mid_u64", "high_u64"]]
    vs = []
    for vf in vfiles:
        v = pd.read_parquet(C.P02 / vf)
        vs.append(v[v.ok & (v.variant != "orig")][["dataset", "file", "variant", "mid_u64", "high_u64"]])
    v = pd.concat(vs, ignore_index=True)
    v = v[v.file.isin(set(o.file))]
    idx = pd.concat([o, v], ignore_index=True).drop_duplicates(["dataset", "file", "variant"])
    idx["is_orig"] = idx.variant == "orig"
    return idx.sort_values(["file", "is_orig", "variant"], ascending=[True, False, True]).reset_index(drop=True)
FILES_PER_ROUND = 12


# CPU part for one file: decode, regenerate its stored edits, re-hash with DBpHash, SSCD preprocessing
def prep_file(f, ix, I, root, D, HC, TR, v3names):
    img = HC.load_bgr(str(root / f))
    if img is None:
        raise RuntimeError(f"cannot read {root / f}")
    seed = HC.variant_seed(f)
    need = I.variant.to_numpy()[ix]
    prim = {}
    if any(n in D.ATTACKS for n in need):
        rng = np.random.default_rng(seed)
        for a in D.ATTACKS:
            prim[a] = D.attack(img, a, rng, HC.CFG["scale_mode"], HC.CFG["rot_mode"])
    out = []
    for r in ix:
        vname = I.variant.iat[r]
        if vname == "orig":
            im = img
        elif vname in prim:
            im = prim[vname]
        elif vname in v3names:
            im = TR.apply2(img, vname, np.random.default_rng(seed))
        else:
            raise ValueError(f"unknown variant {vname}")
        h = D.dbphash(im, HC.CFG["bands"], HC.CFG["interp"])
        ok = HC.bits_to_u64(h["mid"]) == int(I.mid_u64.iat[r]) and HC.bits_to_u64(h["high"]) == int(I.high_u64.iat[r])
        out.append((r, vname, ok, C.sscd_tensor(im)))
    return (f, out)


def main(args, out, holder):
    from concurrent.futures import ThreadPoolExecutor
    D, HC, TR = C.runtime_modules()
    info = json.loads((C.WEIGHTS.parent / "weights_info.json").read_text(encoding="utf-8"))
    pools = ["nft"] if args.smoke else list(POOLS)
    idxs = {n: pool_index(n) for n in pools}
    if args.smoke:
        keep = sorted(idxs["nft"].file.unique())[:20]
        idxs["nft"] = idxs["nft"][idxs["nft"].file.isin(keep)].reset_index(drop=True)
    p = C.Progress("sscd_embeddings", "SSCD 임베딩(원본·편집 재생성)", {n: len(i) for n, i in idxs.items()}, args.smoke, args.log)
    holder["progress"] = p
    model = C.sscd_load()
    rt = C.sscd_runtime_info()
    v3names = set(TR.BY_NAME)
    workers = int(os.environ.get("WORKERS", "0")) or max(2, int((os.cpu_count() or 4) * 0.7))
    report = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name in pools:
            folder = C.OUT_ROOT / "data" / "sscd_emb" / (name + "_smoke") if args.smoke else C.EMB / name
            st = C.EmbStore(folder, idxs[name], info["dim"])
            I = st.index
            root = C.ROOTS[name]
            todo_files = list(I.file[~st.done].unique())
            p.advance(name, int(st.done.sum()), pool=name, resumed=int(st.done.sum()), device=rt["device"])
            mism = {"orig": 0, "variant": 0, "checked": 0, "unreadable": 0, "by_variant": {}}
            rows_by_file = {f: g.index.to_numpy() for f, g in I.groupby("file")}
            rounds = [todo_files[k:k + FILES_PER_ROUND] for k in range(0, len(todo_files), FILES_PER_ROUND)]
            submit = lambda fs: [ex.submit(prep_file, f, rows_by_file[f], I, root, D, HC, TR, v3names) for f in fs]
            pending = submit(rounds[0]) if rounds else []
            for k in range(len(rounds)):
                done_now = [fu.result() for fu in pending]
                pending = submit(rounds[k + 1]) if k + 1 < len(rounds) else []
                items = [it for _, lst in done_now for it in lst]
                E = C.sscd_embed_tensors(model, [it[3] for it in items])
                for (r, vname, ok, _), e in zip(items, E):
                    st.emb[r] = e
                    st.done[r] = True
                    mism["checked"] += 1
                    if not ok:
                        mism["orig" if vname == "orig" else "variant"] += 1
                        mism["by_variant"][vname] = mism["by_variant"].get(vname, 0) + 1
                p.advance(
                    name,
                    len(items),
                    pool=name,
                    file=done_now[-1][0],
                    mismatched_orig=mism["orig"],
                    mismatched_variant=mism["variant"],
                    checked=mism["checked"],
                    device=rt["device"],
                )
                if k % 10 == 0:
                    st.save_done()
            st.save_done()
            report[name] = {"rows": len(I), **mism}
    report["_runtime"] = rt
    total = sum((r["checked"] for k, r in report.items() if k != "_runtime"))
    bad = sum((r["orig"] + r["variant"] for k, r in report.items() if k != "_runtime"))
    return {
        "pools": {k: {kk: v[kk] for kk in ("rows", "orig", "variant", "checked")} for k, v in report.items() if k != "_runtime"},
        "hash_mismatch_rate_in_this_run": bad / total if total else None,
        "runtime": rt,
    }
if __name__ == "__main__":
    C.run_step("sscd_embeddings", "SSCD 임베딩(원본·편집 재생성)", main)
