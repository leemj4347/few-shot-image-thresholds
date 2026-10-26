# Detection rate by edit type
from __future__ import annotations
import pandas as pd
import experiment as C

ALPHAS = (0.01, 0.05, 0.20)


def main(args, out, holder):
    edits = [e for g in C.EDIT_GROUPS.values() for e in g]
    assert len(edits) == 23
    alphas = (0.01,) if args.smoke else ALPHAS
    p = C.Progress("edit_detection", "편집 종류별 검출률", {"read": len(alphas), "aggregate": 1}, args.smoke, args.log)
    holder["progress"] = p
    frames = []
    H = C.hash_convention_rows(args.smoke)
    for a in alphas:
        A = C.read_arms(100, a, modes=("base_set", "base_summary256"))
        if H is not None: A = pd.concat([A, H[(H.n == 100) & (H.alpha == a)]], ignore_index=True)
        frames.append(A)
        p.advance("read", alpha=a)
    A = pd.concat(frames, ignore_index=True)
    cols = ["TPR_" + e for e in edits]
    cg = A.groupby(C.K, dropna=False)[cols].mean().reset_index()
    per_seed = cg.groupby(["method", "alpha", "training_seed"])[cols].mean().reset_index()
    per_edit = per_seed.groupby(["method", "alpha"])[cols].mean().reset_index()
    long = per_edit.melt(id_vars=["method", "alpha"], var_name="edit", value_name="TPR")
    long["edit"] = long.edit.str.replace("TPR_", "", regex=False)
    group_of = {e: g for g, es in C.EDIT_GROUPS.items() for e in es}
    long["group"] = long.edit.map(group_of)
    C.save_csv(long, out / "per_edit_tpr.csv")
    grp = long.groupby(["method", "alpha", "group"]).TPR.mean().unstack("group").reset_index()
    C.save_csv(grp, out / "per_group_tpr.csv")
    wide = long.pivot_table(index=["group", "edit"], columns=["method", "alpha"], values="TPR")
    wide.to_csv(C.output_path(out / "per_edit_wide.csv"), encoding="utf-8-sig")
    p.advance("aggregate")
    return {"methods": int(long.method.nunique()), "hash_convention_included": H is not None,
            "groups": list(C.EDIT_GROUPS)}


if __name__ == "__main__":
    C.run_step("edit_detection", "편집 종류별 검출률", main)
