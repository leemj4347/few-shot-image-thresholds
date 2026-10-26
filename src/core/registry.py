# Registry pools, growth and query sampling
from __future__ import annotations
import numpy as np
import pandas as pd
import similarity as SIM


SCHEMES = {"dbp": ("mid_u64", "high_u64", SIM.W_FIXED),
           "phash": ("phash_u64", None, SIM.W_SINGLE),
           "iphash": ("high_u64", None, SIM.W_SINGLE)}


# Pool of representative originals from one or more cache parquet files
class Pool:
    def __init__(self, parquets, name: str | None = None, collections: list[str] | None = None, scheme: str = "dbp"):
        if isinstance(parquets, (str, bytes)) or hasattr(parquets, "__fspath__"):
            parquets = [parquets]
        df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
        df = df[df.ok & (df.variant == "orig") & df.is_rep].copy()
        if collections is not None:
            df = df[df.collection.isin(collections)]
        df = df.sort_values(["dataset", "file"]).reset_index(drop=True)

        df = df.drop_duplicates("group_id", keep="first").reset_index(drop=True)
        self.df = df
        self.name = name or "+".join(sorted(df.dataset.unique()))
        self.scheme = scheme
        mcol, hcol, self.w = SCHEMES[scheme]
        self.mid = df[mcol].to_numpy(dtype=np.uint64)
        self.high = df[hcol].to_numpy(dtype=np.uint64) if hcol else np.zeros(len(df), dtype=np.uint64)
        self.coll = df.collection.fillna("").to_numpy(dtype=object)
        self.has_coll = bool((self.coll != "").any())
        self.n = len(df)

    def collections(self) -> np.ndarray:
        return np.array(sorted(set(self.coll[self.coll != ""])), dtype=object)

    def subset(self, idx: np.ndarray):
        return self.mid[idx], self.high[idx]


class Registry:
    def __init__(self, pool: Pool, idx: np.ndarray, structure: str, seed: int, M_target: int, rule: dict | None = None,
                 exclude_coll: set | None = None):
        self.pool = pool
        self.idx = np.asarray(idx, dtype=np.int64)
        self.structure = structure
        self.seed = seed
        self.M_target = M_target
        self.rule = rule or {}
        self.exclude_coll = set(exclude_coll or ())
        self.mid, self.high = pool.subset(self.idx)
        self.M = len(self.idx)
        self.used_coll = set(pool.coll[self.idx]) - {""}


    @classmethod
    def sample(cls, pool: Pool, M: int, seed: int, structure: str = "random",
               exclude_coll: set | None = None, exclude_idx: np.ndarray | None = None):
        rng = np.random.default_rng(seed)
        avail = np.ones(pool.n, dtype=bool)
        if exclude_coll: avail &= ~np.isin(pool.coll, list(exclude_coll))
        if exclude_idx is not None: avail[exclude_idx] = False
        cand = np.flatnonzero(avail)
        if structure == "random" or not pool.has_coll:
            if len(cand) < M: raise ValueError(f"pool {pool.name}: {len(cand)} < M={M}")
            idx = rng.choice(cand, M, replace=False)
            return cls(
                pool,
                np.sort(idx),
                "random" if not pool.has_coll else structure,
                seed,
                M,
                exclude_coll=exclude_coll,
            )
        colls = np.array(sorted(set(pool.coll[cand]) - {""}), dtype=object)
        rng.shuffle(colls)
        by = {c: rng.permutation(cand[pool.coll[cand] == c]) for c in colls}
        chosen = []
        if structure == "concentrated":
            for c in colls:
                need = M - len(chosen)
                if need <= 0: break
                chosen.extend(by[c][:need].tolist())
        elif structure == "spread":
            ptr = {c: 0 for c in colls}
            while len(chosen) < M:
                progressed = False
                for c in colls:
                    if len(chosen) >= M: break
                    if ptr[c] < len(by[c]):
                        chosen.append(int(by[c][ptr[c]]))
                        ptr[c] += 1
                        progressed = True
                if not progressed: break
        else:
            raise ValueError(structure)
        if len(chosen) < M: raise ValueError(f"pool {pool.name}: only {len(chosen)} for M={M} ({structure})")
        return cls(
            pool,
            np.sort(np.array(chosen)),
            structure,
            seed,
            M,
            {"coll_order": list(colls)},
            exclude_coll=exclude_coll,
        )


    # Put at least k originals with variant hashes into the registry
    def ensure_files(self, files: set, k: int, seed: int | None = None):
        if k <= 0 or not files: return self
        pool = self.pool
        fmask = pool.df.file.isin(files).to_numpy()
        cur = self.idx
        need = int(k) - int(fmask[cur].sum())
        if need <= 0: return self
        rng = np.random.default_rng((self.seed if seed is None else seed) + 104729)
        new_idx = cur.copy()
        in_reg = set(new_idx.tolist())
        groups = sorted(set(pool.coll[cur])) if pool.has_coll else [None]
        for c in groups:
            if need <= 0: break
            sel = (pool.coll[new_idx] == c) if c is not None else np.ones(len(new_idx), bool)
            in_c = new_idx[sel]
            drop_pool = in_c[~fmask[in_c]]
            cand_mask = fmask & ~np.isin(np.arange(pool.n), new_idx)
            if c is not None: cand_mask &= (pool.coll == c)
            cand = np.flatnonzero(cand_mask)
            m = min(need, len(drop_pool), len(cand))
            if m <= 0: continue
            drop = rng.choice(drop_pool, m, replace=False)
            add = rng.choice(cand, m, replace=False)
            new_idx = np.sort(np.concatenate([np.setdiff1d(new_idx, drop), add]))
            in_reg = set(new_idx.tolist())
            need -= m
        if len(new_idx) != len(cur): return self
        return Registry(pool, new_idx, self.structure, self.seed, self.M_target, self.rule, self.exclude_coll)


    # (i) Same composition, larger M
    def grow(self, M_new: int, seed: int | None = None):
        assert M_new >= self.M
        if self.structure.endswith("+new"): raise ValueError("유입본(+new)은 성장 (i)의 출발점이 아님")
        seed = self.seed if seed is None else seed
        if self.structure in ("random",) or not self.pool.has_coll:
            extra = Registry.sample(
                self.pool,
                M_new - self.M,
                seed + 7919,
                "random",
                exclude_coll=self.exclude_coll,
                exclude_idx=self.idx,
            )
            return Registry(
                self.pool,
                np.sort(np.concatenate([self.idx, extra.idx])),
                self.structure,
                self.seed,
                M_new,
                exclude_coll=self.exclude_coll,
            )
        if self.structure == "concentrated":

            rng = np.random.default_rng(seed + 7919)
            chosen = list(self.idx)
            order = self.rule.get("coll_order", list(self.pool.collections()))
            for c in order:
                need = M_new - len(chosen)
                if need <= 0: break
                cand = np.flatnonzero((self.pool.coll == c))
                cand = cand[~np.isin(cand, chosen)]
                rng.shuffle(cand)
                chosen.extend(cand[:need].tolist())
            if len(chosen) < M_new: raise ValueError(f"grow(concentrated): only {len(chosen)} < {M_new}")
            return Registry(
                self.pool,
                np.sort(np.array(chosen)),
                self.structure,
                self.seed,
                M_new,
                self.rule,
                self.exclude_coll,
            )

        rng = np.random.default_rng(seed + 7919)
        chosen = list(self.idx)
        colls = sorted(self.used_coll)
        ptr = {c: rng.permutation(np.flatnonzero((self.pool.coll == c) & ~np.isin(np.arange(self.pool.n), self.idx))) for c in colls}
        pos = {c: 0 for c in colls}
        while len(chosen) < M_new:
            progressed = False
            for c in colls:
                if len(chosen) >= M_new: break
                if pos[c] < len(ptr[c]):
                    chosen.append(int(ptr[c][pos[c]]))
                    pos[c] += 1
                    progressed = True
            if not progressed: break
        if len(chosen) < M_new: raise ValueError(f"grow(spread): only {len(chosen)} < {M_new}")
        return Registry(
            self.pool,
            np.sort(np.array(chosen)),
            self.structure,
            self.seed,
            M_new,
            self.rule,
            self.exclude_coll,
        )

    # (ii) Inflow of new collections
    def add_collections(self, n_new_coll: int, seed: int, per_coll: int | None = None, allowed: set | None = None):
        rng = np.random.default_rng(seed + 104729)
        if allowed:
            rest = [c for c in self.pool.collections() if c in allowed and c not in self.used_coll]
        else:
            rest = [c for c in self.pool.collections() if c not in self.used_coll and c not in self.exclude_coll]
        if len(rest) < n_new_coll: raise ValueError(f"add_collections: only {len(rest)} unused collections < {n_new_coll}")
        rng.shuffle(rest)
        chosen = list(self.idx)
        for c in rest[:n_new_coll]:
            cand = rng.permutation(np.flatnonzero(self.pool.coll == c))
            chosen.extend(cand[:per_coll].tolist() if per_coll else cand.tolist())
        return Registry(
            self.pool,
            np.sort(np.array(chosen)),
            self.structure + "+new",
            self.seed,
            len(chosen),
            self.rule,
            self.exclude_coll,
        )


    def query_max(self, qm, qh, topk=0, hist=False, chunk=None):
        return SIM.query_max(np.asarray(qm, dtype=np.uint64), np.asarray(qh, dtype=np.uint64), self.mid, self.high,
                             w=self.pool.w, chunk=chunk, topk=topk, hist=hist)

    # Top-k pair review list for normal queries
    def verify_list(self, qm, qh, qfiles, k=5, tau=None) -> pd.DataFrame:
        r = self.query_max(qm, qh, topk=k)
        rows = []
        for i in range(len(qm)):
            if tau is not None and r["Z"][i] < tau - SIM.TOL: continue
            for j in range(k):
                ri = r["topk_idx"][i, j]
                if ri < 0: continue
                rows.append({"query": qfiles[i], "rank": j + 1, "registry_file": self.pool.df.file.iloc[self.idx[ri]],
                             "registry_collection": self.pool.coll[self.idx[ri]], "S": float(r["topk_sim"][i, j])})
        return pd.DataFrame(rows)

    def files(self) -> np.ndarray:
        return self.pool.df.file.to_numpy()[self.idx]

    def n_collections(self) -> int:
        return len(self.used_coll)


# Per seed, hold out a fraction of collections for queries only
def holdout_collections(pool: Pool, seed: int, frac: float = 0.3, min_reps: int = 50,
                        need: int | None = None) -> set:
    colls = pool.collections()
    if len(colls) == 0: return set()
    cnt = pd.Series(pool.coll).value_counts()
    colls = np.array([c for c in colls if cnt.get(c, 0) >= min_reps], dtype=object)
    if len(colls) == 0: return set()
    rng = np.random.default_rng(seed + 2654435761)
    perm = rng.permutation(colls)
    if need is None:
        k = max(1, int(np.ceil(frac * len(colls))))
    else:


        k, acc = 0, 0
        while k < len(perm) and acc < need:
            acc += int(cnt.get(perm[k], 0))
            k += 1
        k = max(1, k)
    k = min(k, max(0, len(colls) - 1))
    return set(perm[:k].tolist())


# Collections reserved for inflow (condition ii)
def reserve_collections(pool: Pool, seed: int, n_new: int, hold: set, min_reps: int = 50) -> set:
    if not pool.has_coll or n_new <= 0: return set()
    cnt = pd.Series(pool.coll).value_counts()
    cand = sorted(c for c in pool.collections() if c not in hold and cnt.get(c, 0) >= min_reps)
    if len(cand) <= n_new: return set()
    rng = np.random.default_rng(seed + 22801763)
    return set(rng.permutation(np.array(cand, dtype=object))[:n_new].tolist())


# Normal query indices
def sample_queries(pool: Pool, reg: Registry, n: int, seed: int, exclude_idx: np.ndarray | None = None,
                   exclude_files: set | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed + 15485863)
    avail = np.ones(pool.n, dtype=bool)
    avail[reg.idx] = False
    if exclude_idx is not None: avail[exclude_idx] = False
    if exclude_files:
        avail &= ~pool.df.file.isin(exclude_files).to_numpy()
    if pool.has_coll and reg.used_coll:
        avail &= ~np.isin(pool.coll, list(reg.used_coll))
    if pool.has_coll and reg.exclude_coll:

        avail &= (np.isin(pool.coll, list(reg.exclude_coll)) | (pool.coll == ""))
    cand = np.flatnonzero(avail)
    if len(cand) <= n: return np.sort(cand)
    return np.sort(rng.choice(cand, n, replace=False))


# (mid, high) uint64 arrays for the pool scheme from the variant DataFrame
def variant_hashes(pool: Pool, variants: pd.DataFrame):
    mcol, hcol, _ = SCHEMES[pool.scheme]
    m = variants[mcol].to_numpy(dtype=np.uint64)
    h = variants[hcol].to_numpy(dtype=np.uint64) if hcol else np.zeros(len(variants), dtype=np.uint64)
    return m, h


# Variant rows of the registry originals
def positives_for(reg: Registry, variants: pd.DataFrame, max_originals: int | None, seed: int):
    rf = set(reg.files())
    v = variants[variants.file.isin(rf)]
    if max_originals and v.file.nunique() > max_originals:
        rng = np.random.default_rng(seed + 32452843)
        keep = rng.choice(np.array(sorted(v.file.unique())), max_originals, replace=False)
        v = v[v.file.isin(keep)]
    return v.reset_index(drop=True)
