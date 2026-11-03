# Near-duplicate check across roles
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import experiment as C

R_BITS = 3
CHUNKS = 4


def popcnt(x):
    if hasattr(np, "bitwise_count"): return np.bitwise_count(x).astype(np.int16)
    b = np.ascontiguousarray(x).view(np.uint8).reshape(x.shape + (8,))
    return np.unpackbits(b, axis=-1).sum(-1).astype(np.int16)


def main(args, out):
    S, MF, T, REG = C.modules()

    _, m, cfg = S.X.reference_inputs(S.X.DEFAULT)
    root = Path(cfg["root"])
    table = T.load_role_table(m["role_table"])
    parts = []
    for name, pc in cfg["pools"].items():
        pool = REG.Pool(root / pc["cache"], name=name, scheme="dbp")
        keys = T.split_keys(pool)
        roles = T.roles_from_table(pool, table)
        role_of = {k: r for r, ks in roles.items() for k in ks}
        parts.append(pd.DataFrame({"pool": name, "file": pool.df.file.astype(str).to_numpy(),
                                   "mid": pool.mid, "high": pool.high, "role": [role_of[k] for k in keys]}))
    D = pd.concat(parts, ignore_index=True)
    D = D.drop_duplicates(["pool", "file"]).reset_index(drop=True)
    if args.smoke: D = D.sample(min(len(D), 20000), random_state=0).reset_index(drop=True)

    mid = D["mid"].to_numpy(np.uint64)
    high = D["high"].to_numpy(np.uint64)
    role = D.role.to_numpy()
    N = len(D)
    acc = []
    for j in range(CHUNKS):
        key = ((mid >> np.uint64(16 * j)) & np.uint64(0xFFFF)).astype(np.int64)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        cuts = np.flatnonzero(np.diff(ks)) + 1
        for grp in np.split(order, cuts):
            if len(grp) < 2: continue
            gm, gh, gr = mid[grp], high[grp], role[grp]
            for s0 in range(0, len(grp), 2048):
                a = slice(s0, s0 + 2048)
                dm = popcnt(gm[a, None] ^ gm[None, :])
                dh = popcnt(gh[a, None] ^ gh[None, :])
                ii, jj = np.nonzero((dm <= R_BITS) & (dh <= R_BITS) & (gr[a, None] != gr[None, :]))
                if len(ii):
                    x = grp[ii + s0].astype(np.int64)
                    y = grp[jj].astype(np.int64)
                    keep = x < y
                    acc.append(np.stack([x[keep], y[keep], dm[ii, jj][keep], dh[ii, jj][keep]], 1))
        n_so_far = int(sum(len(t) for t in acc))

    if acc:
        arr = np.concatenate(acc)
        code = arr[:, 0] * N + arr[:, 1]
        _, first = np.unique(code, return_index=True)
        arr = arr[first]
    else:
        arr = np.zeros((0, 4), np.int64)
    P = pd.DataFrame(arr, columns=["i", "j", "d_mid", "d_high"])
    if len(P):
        for side in ("i", "j"):
            P[f"pool_{side}"] = D.pool.to_numpy()[P[side]]
            P[f"file_{side}"] = D.file.to_numpy()[P[side]]
            P[f"role_{side}"] = D.role.to_numpy()[P[side]]
        P["identical"] = (P.d_mid == 0) & (P.d_high == 0)
        P["role_pair"] = [" / ".join(sorted(t)) for t in zip(P.role_i, P.role_j)]
    C.save_csv(P.head(100000), out / "cross_role_pairs.csv")
    summ = (P.groupby(["role_pair", "identical"]).size().rename("pairs").reset_index()
            if len(P) else pd.DataFrame(columns=["role_pair", "identical", "pairs"]))
    C.save_csv(summ, out / "cross_role_summary.csv")
    imgs = {}
    if len(P):
        for r in ("train", "dev", "test"):
            sel = np.r_[P.i[P.role_i == r].to_numpy(), P.j[P.role_j == r].to_numpy()]
            imgs[r] = int(len(np.unique(sel)))

    return {"images": len(D), "cross_role_pairs": len(P), "identical_pairs": int(P.identical.sum()) if len(P) else 0,
            "images_involved_by_role": imgs, "r_bits": R_BITS}


if __name__ == "__main__":
    C.run_step("data_overlap", "역할 간 근사 중복 점검", main)
