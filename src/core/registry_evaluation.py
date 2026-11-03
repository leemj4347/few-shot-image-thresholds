# Construct and evaluate registry task bundles
from __future__ import annotations
import argparse, hashlib, json, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import similarity as SIM

import criteria as CR
import evaluate_registry as EV
import tasks as T
from registry import Pool
from methods import fixed as MF, empirical as ME, statmlp as MS, devtable as MD

MAIN_N = 100
MIN_TASKS = 36
MIN_DET_TASKS = 10


TPR_REQ_LEGACY = {0.05: 0.85, 0.01: 0.70}


COND_KEYS = ["domain", "structure", "kind", "M", "alpha"]

TASK_KEYS = COND_KEYS + ["scheme", "seed", "n"]


# Exact binomial confidence interval (Clopper and Pearson, 1934)
def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0: return (float("nan"), float("nan"))
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


# False alarm rate interval for one task
def far_interval(far: float, n_eval: int, conf: float = 0.95) -> tuple[float, float]:
    return clopper_pearson(int(round(far * n_eval)), int(n_eval), conf)


# Exact McNemar test on discordant violation counts of the same tasks
def mcnemar_exact(a_viol: np.ndarray, b_viol: np.ndarray) -> dict:
    n01 = int(np.sum(~a_viol & b_viol))
    n10 = int(np.sum(a_viol & ~b_viol))
    n = n01 + n10
    p = float(binomtest(n10, n, 0.5).pvalue) if n else float("nan")
    return {"n_pairs": int(len(a_viol)), "only_A_violates": n10, "only_B_violates": n01, "p_value": p}


# Task bundles
def build_bundles(cfg: dict, root: Path, role_table: dict, scheme: str, log=print,
                  invalid: list | None = None) -> list:
    if invalid is None: invalid = []
    pools, variants = {}, {}
    for pool_index, (name, p) in enumerate(cfg["pools"].items(), 1):

        pools[name] = Pool(root / p["cache"], name=name, scheme=scheme)
        frames = []
        for key in ("variants", "variants2"):
            rel = p.get(key)
            if rel is None and key == "variants2" and p.get("variants"):
                rel = str(p["variants"]).replace("_var.parquet", "_var2.parquet")
            if rel and (root / rel).exists():
                v = pd.read_parquet(root / rel)
                frames.append(v[v.ok & (v.variant != "orig")])
        if frames: variants[name] = pd.concat(frames, ignore_index=True)

    out = []
    for dom, d in cfg["domains"].items():
        names = d["pools"]
        pool = pools[names[0]] if len(names) == 1 else Pool([root / cfg["pools"][n]["cache"] for n in names],
                                                           name=dom, scheme=scheme)
        var = pd.concat([variants[n] for n in names if n in variants], ignore_index=True) \
            if any(n in variants for n in names) else None
        roles = T.roles_from_table(pool, role_table)
        for role in T.ROLES:

            if role == "train" and scheme != "dbp": continue
            rp = T.role_pool(pool, roles, role)
            if rp.n < 50: continue

            M_list = cfg.get("train_M", d["M"]) if role == "train" else d["M"]
            structures=[st for st in d.get("structures", ["random"]) if st=="random" or rp.has_coll]
            total_jobs=len(structures)*len(M_list)*len(cfg["seeds"])
            done_jobs=0
            invalid_jobs=0
            for struct in structures:
                for M in M_list:
                    for seed in cfg["seeds"]:

                        b = T.build_bundle(rp, var, role, dom, struct, M, seed, cfg)
                        done_jobs+=1
                        if b.note:invalid_jobs+=1

                        if b.note:
                            invalid.append({"role": role, "domain": dom, "structure": struct,
                                            "M": int(M), "seed": int(seed), "reason": b.note})
                            continue
                        out.append(b)
        n_bad = sum(1 for r in invalid if r["domain"] == dom)
        log(f"  {dom}: 태스크 {sum(1 for b in out if b.domain == dom)}개, 무효 {n_bad}개")
    return out


# Split meta-training tasks into fit and calibration sets (seeded, deterministic)
def split_train_cal(train_bundles: list, frac: float = 0.7) -> tuple[list, list]:
    seeds = sorted({b.seed for b in train_bundles})
    cut = max(1, int(round(len(seeds) * frac)))
    fit_seeds = set(seeds[:cut])
    cal_seeds = set(seeds[cut:])
    if not cal_seeds or fit_seeds & cal_seeds: raise ValueError("disjoint fit/cal seeds required")
    return ([b for b in train_bundles if b.seed in fit_seeds],
            [b for b in train_bundles if b.seed in cal_seeds])


def score_all(bundles: list, cfg: dict, scheme: str, included: dict | None, log=print) -> tuple[pd.DataFrame, dict]:
    alphas = cfg["alphas"]
    n_list = [n for n in T.N_LIST if n <= cfg.get("n_max", T.N_MAX)]
    w = SIM.SCHEME_W[scheme]
    grid = SIM.grid_values(w)
    byrole = {r: [b for b in bundles if b.role == r] for r in T.ROLES}
    if not byrole["test"]:
        raise ValueError("no test-role tasks in the bundle set")
    dev = byrole["dev"]

    d0, gen = MF.dev_pair_scores(dev) if dev else (None, np.zeros(0))
    fixed_taus = {}
    if dev:
        fixed_taus["개발 최적 고정"] = MF.fit_C0g(dev)
        fixed_taus["등오류율 고정"] = MF.fit_C0eer(d0, gen, grid)
        fixed_taus["Youden J 고정"] = MF.fit_C0j(d0, gen, grid)
        fixed_taus["Otsu 고정"] = MF.fit_C0otsu(d0, gen, grid)
    fixed_taus["원문 식 28"] = SIM.grid_ceil(MF.GAMMA_REF, w) if scheme == "dbp" else float("nan")
    fixed_taus["해밍 관행값"] = MF.fit_C0h(w)
    dom_taus = {dom: MF.fit_C1(dev, dom) for dom in cfg["domains"]} if dev else {}


    devfit = MD.fit(dev, alphas, w) if dev else None

    info = {"fixed_taus": {k: (None if not np.isfinite(v) else float(v)) for k, v in fixed_taus.items()}}
    if devfit is not None:
        info["dev_constant"] = {str(k): (None if not np.isfinite(v) else float(v))
                                for k, v in devfit["constant"].items()}
        info["dev_table_size"] = int(sum(1 for v in devfit["table"].values() if np.isfinite(v)))
        info["dev_table"] = {"|".join(map(str, k)): (None if not np.isfinite(v) else float(v))
                             for k, v in sorted(devfit["table"].items(), key=lambda kv: str(kv[0]))}
    model, lam = None, {}
    if scheme == "dbp" and byrole["train"]:
        fitb, calb = split_train_cal(byrole["train"], cfg.get("cal_frac", 0.7))


        need_cal = int(cfg.get("min_cal_tasks", 0))
        if need_cal and len(calb) < need_cal:
            sys.exit(f"보정 태스크 {len(calb)}개 < 하한 {need_cal}개. "
                     f"메타훈련 태스크 {len(byrole['train'])}개, 분할 {cfg.get('cal_frac', 0.7)}.\n"
                     "시드 수나 메타훈련 M 격자를 늘리거나 분할 비율을 바꿀 것. "
                     "Λ 보정의 하한은 alpha=0.05에서 19개, alpha=0.01에서 99개이며 여유를 두어야 한다.")

        model, minfo = MS.fit(fitb, alphas, n_list, seed=cfg.get("control_seed", 0),
                              hidden=cfg.get("control_hidden", MS.HIDDEN), epochs=cfg.get("control_epochs", MS.EPOCHS),
                              progress=None)

        info["control_model"] = minfo
        info["control_tasks"] = {"fit": len(fitb), "cal": len(calb)}
        if model is not None:
            cal_done=0
            cal_total=len(alphas)*len(n_list)
            for a in alphas:
                for n in n_list:

                    lam[(a, n)] = MS.calibrate(model, calb, a, n)
                    cal_done+=1

            info["Lambda"] = {f"alpha={a},n={n}": v for (a, n), v in lam.items()}
        log(f"  대조군: {minfo}")

    rows = []
    for test_index,b in enumerate(byrole["test"],1):

        for kind, d in b.reg.items():
            if len(d["Z_eval"]) == 0: continue
            M_eff = d["M"]
            n_cand = len(d["Z_sup"]) + len(d["Z_eval"])
            for a in alphas:
                tau_s = MF.tau_C0s(d0, M_eff, a) if d0 is not None else float("nan")
                for n in n_list:
                    if n > b.n_sup: continue
                    cand = dict(fixed_taus)
                    table_src = None
                    cand["도메인별 재조정"] = dom_taus.get(b.domain, float("nan"))
                    cand["전역 규모 보정"] = tau_s
                    if devfit is not None:
                        cand["개발 자료 고정값"] = MD.tau_constant(devfit, a)
                        tv, src = MD.tau_table(devfit, b.domain, b.structure, kind, int(b.M), a)
                        cand["개발 자료 조견표"] = tv
                        table_src = src
                    cand["NP 최댓값"] = ME.tau_C2max(d["Z_sup"], n, a, w)
                    cand["경험 분위수"] = ME.tau_C2emp(d["Z_sup"], n, a, w)
                    cand["NP 비복원"] = ME.tau_C2max_fp(d["Z_sup"], n, a, w, n_cand)
                    hs = d["hist_sup"].get(n)
                    if hs is not None:
                        tp, okp, need = ME.tau_C2pair(hs, M_eff, a, w)
                        cand["NP 쌍 점수"] = tp
                    else:
                        okp, need = False, -1
                    if model is not None:

                        cand["요약 통계량 MLP"] = MS.tau(model, lam.get((a, n), {}).get("Lambda", float("inf")),
                                                    d["Z_sup"], n, M_eff, a, w)
                    for meth, tau in cand.items():
                        if isinstance(tau, float) and np.isnan(tau): continue
                        r = EV.score(tau, d, a)
                        lo, hi = far_interval(r["FAR"], r["n_eval"])
                        band = "A" if lo > a else ("B" if hi < a else "U")
                        r.update({"scheme": scheme, "method": meth, "kind": kind, "alpha": a, "n": n,
                                  "domain": b.domain, "structure": b.structure, "M": b.M, "M_eff": M_eff,
                                  "seed": b.seed, "n_sup": b.n_sup, "n_cand": n_cand,
                                  "FAR_lo": lo, "FAR_hi": hi, "band": band,
                                  "table_source": table_src if meth == "개발 자료 조견표" else None,
                                  "np_pair_ok": okp if meth == "NP 쌍 점수" else None,
                                  "np_pair_need": need if meth == "NP 쌍 점수" else None})
                        if included is not None:
                            det, n_inc, n_miss = detection_included(r, included, b, kind, a)
                            r["TPR_included"] = det
                            r["TPR_included_n"] = n_inc
                            r["TPR_included_missing"] = n_miss
                        rows.append(r)

    return pd.DataFrame(rows), info


# Detection rate metrics
def detection_included(r: dict, included: dict, b, kind: str, alpha: float) -> tuple[float, int, int]:
    key = (b.domain, b.structure, kind, int(b.M), float(alpha))
    names = included.get(key)
    if not names: return (float(r.get("TPR_photo3", float("nan"))), 0, 0)
    vals, miss = [], 0
    for v in names:
        col = f"TPR_{v}"
        if col in r and not (isinstance(r[col], float) and np.isnan(r[col])):
            vals.append(float(r[col]))
        else:
            miss += 1
    if miss: return (float("nan"), len(names), miss)
    return (float(np.mean(vals)), len(names), 0)


# Included edits with no column in the scored data
def missing_included_columns(df: pd.DataFrame, included: dict) -> list:
    if not len(df) or not included: return []
    want = sorted({v for names in included.values() for v in names})
    have = {c[4:] for c in df.columns if c.startswith("TPR_")}
    return [v for v in want if v not in have]


# inclusion_fixed.json -> {(domain, structure, kind, M, alpha): [edit names]}
def load_included(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists(): return None
    fixed = json.load(open(path, encoding="utf-8"))
    out = {}
    for c in fixed.get("conditions", []):
        out[(c["domain"], c["structure"], c["kind"], int(c["M"]), float(c["alpha"]))] = list(c.get("included", []))
    return out


# Add the 23-edit mean detection rate and balanced accuracy per row, plus the task baseline BA
def add_ba(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["TPR_" + v for v in CR.variant_names()]
    d = df.copy()
    have = [c for c in cols if c in d.columns]
    if len(have) < len(cols):
        d["TPR23"] = np.nan
        d["missing_variants"] = len(cols) - len(have)
    else:
        d["TPR23"] = d[cols].mean(axis=1)
        d["missing_variants"] = d[cols].isna().sum(axis=1).astype(int)
    d["BA23"] = (d["TPR23"] + 1.0 - d["FAR"]) / 2.0
    keys = [k for k in TASK_KEYS if k in d.columns]
    base = d[d.method == CR.BASELINE_METHOD][keys + ["BA23"]].rename(columns={"BA23": "base_BA23"})
    base = base.drop_duplicates(keys)
    return d.merge(base, on=keys, how="left", validate="many_to_one")


def condition_table(df: pd.DataFrame, n_main: int = MAIN_N) -> pd.DataFrame:
    df = add_ba(df)
    d = df[df.n == n_main]
    rows = []
    for key, s in d.groupby(COND_KEYS + ["scheme", "method"], dropna=False):
        cond = dict(zip(COND_KEYS + ["scheme", "method"], key))
        T_ = len(s)
        D = int(s.violation.sum())
        lo, hi = clopper_pearson(D, T_)


        det_col = "TPR_included" if "TPR_included" in s else "TPR_photo3"
        ok = s[~s.violation]


        n_det = int(ok[det_col].notna().sum())
        n_det_missing = int(ok[det_col].isna().sum())

        sb = s[s.BA23.notna() & s.base_BA23.notna()]
        V_ = D / T_ if T_ else np.nan
        BA_ = float(sb.BA23.mean()) if len(sb) else np.nan
        BAb_ = float(sb.base_BA23.mean()) if len(sb) else np.nan
        TPR_ = float(sb.TPR23.mean()) if len(sb) else np.nan
        ok_far = bool(V_ <= CR.DELTA + 1e-12) if np.isfinite(V_) else False
        ok_perf = bool(BA_ >= BAb_ - 1e-12) if (np.isfinite(BA_) and np.isfinite(BAb_)) else False
        rows.append(dict(cond, n_tasks=T_, tasks_enough=T_ >= MIN_TASKS, D=D, V=V_,
                         TPR23=TPR_, BA23=BA_, baseline_BA23=BAb_,
                         n_tasks_ba=int(len(sb)),
                         meets_far=ok_far, meets_perf=ok_perf,
                         meets=bool(ok_far and ok_perf),
                         no_detection=bool(ok_far and ok_perf and np.isfinite(TPR_) and TPR_ <= 0.0),
                         n_tasks_detection=n_det, detection_missing_tasks=n_det_missing,
                         detection_judgeable=(n_det_missing == 0) and (n_det >= MIN_DET_TASKS),
                         V_lo=lo, V_hi=hi,
                         excess_mean_violating=float(s.loc[s.violation, "excess"].mean()) if D else 0.0,
                         detection_ok=float(ok[det_col].mean()) if len(ok) else np.nan,
                         band_A=int((s.band == "A").sum()), band_U=int((s.band == "U").sum()),
                         band_B=int((s.band == "B").sum()),
                         sentinel_rate=float(s.sentinel.mean()),
                         n_eval_median=float(s.n_eval.median()),
                         M_eff_median=float(s.M_eff.median()),
                         M_eff_min=int(s.M_eff.min()), M_eff_max=int(s.M_eff.max())))
    return pd.DataFrame(rows)


def mcnemar_table(df: pd.DataFrame, n_main: int = MAIN_N) -> pd.DataFrame:
    d = df[(df.n == n_main)]
    rows = []
    for key, s in d.groupby(COND_KEYS + ["scheme"], dropna=False):
        piv = s.pivot_table(index="seed", columns="method", values="violation")
        for m1, m2 in combinations(sorted(piv.columns), 2):
            sub = piv[[m1, m2]].dropna()
            if len(sub) < 2: continue
            r = mcnemar_exact(sub[m1].to_numpy(bool), sub[m2].to_numpy(bool))
            rows.append(dict(zip(COND_KEYS + ["scheme"], key), method_A=m1, method_B=m2, **r))
    return pd.DataFrame(rows)


# Stop criterion
def gate(cond: pd.DataFrame, kinds=("base", "grow_i", "grow_ii")) -> dict:
    out = {"rule": dict(CR.rule_dict(), min_tasks=MIN_TASKS, n_main=MAIN_N, kinds=list(kinds)),
           "methods": {}, "verdict": None, "shortfall": [], "unjudgeable": []}
    if len(cond) == 0:
        out["verdict"] = "자료 없음"
        out["n_unjudgeable"] = 0
        return out
    d = cond[(cond.scheme == "dbp") & cond.kind.isin(kinds)]
    for meth, s in d.groupby("method"):
        fails, undecided, detail, n_nodet = 0, 0, [], 0
        for _, r in s.iterrows():
            cond_id = {"domain": r.domain, "structure": r.structure, "kind": r.kind,
                       "M": int(r.M), "alpha": float(r.alpha)}

            if not np.isfinite(r.BA23) or not np.isfinite(r.baseline_BA23):
                undecided += 1
                out["unjudgeable"].append(dict(cond_id, method=meth, n_tasks=int(r.n_tasks),
                                               n_tasks_ba=int(r.get("n_tasks_ba", 0))))
                continue
            if bool(r.no_detection):
                n_nodet += 1
            if bool(r.meets):
                continue
            fails += 1
            which = ([] if bool(r.meets_far) else ["오경보"]) + ([] if bool(r.meets_perf) else ["성능"])
            detail.append(dict(cond_id, M_eff_median=float(r.M_eff_median),
                               V=float(r.V), V_hi=float(r.V_hi),
                               BA=float(r.BA23), baseline_BA=float(r.baseline_BA23),
                               TPR=None if not np.isfinite(r.TPR23) else float(r.TPR23),
                               tasks=int(r.n_tasks), tasks_enough=bool(r.tasks_enough),
                               fail=which))
        ok_all = (fails == 0 and undecided == 0)
        enough = s[s.tasks_enough]
        ok_enough = True
        for _, r in enough.iterrows():
            ok_enough &= bool(np.isfinite(r.BA23) and np.isfinite(r.baseline_BA23) and bool(r.meets))
        out["methods"][meth] = {"n_conditions": int(len(s)),
                                "n_met": int(s.meets.sum()),
                                "n_met_no_detection": int(n_nodet),
                                "pass": bool(ok_all),
                                "n_failed": int(fails), "n_undecided": int(undecided),
                                "n_conditions_enough_tasks": int(len(enough)),
                                "pass_among_enough_tasks": bool(ok_enough) if len(enough) else None,
                                "shortfall": detail[:20]}
    out["n_unjudgeable"] = len(out["unjudgeable"])
    passed = [m for m, v in out["methods"].items() if v["pass"]]
    held = sorted([m for m, v in out["methods"].items() if v["n_failed"] == 0 and v["n_undecided"] > 0])
    if held:
        out["verdict"] = ("판정 보류: " + ", ".join(held) + " 은(는) 떨어진 조건이 없으나 "
                          + "성능 요건을 읽을 수 없는 조건이 남아 있다("
                          + ", ".join(f"{m} {out['methods'][m]['n_undecided']}건" for m in held) + ")")
    elif passed:
        out["verdict"] = "충족: " + ", ".join(passed)
    else:
        out["verdict"] = "미충족: 미달 조건을 다음 평가의 개발 목표로 둔다"
        for key, s in d.groupby(COND_KEYS):
            s2 = s.sort_values("V")
            best = s2.iloc[0]
            out["shortfall"].append({"condition": dict(zip(COND_KEYS, key)),
                                     "closest_method": best.method, "V": float(best.V),
                                     "BA": None if not np.isfinite(best.BA23) else float(best.BA23),
                                     "baseline_BA": None if not np.isfinite(best.baseline_BA23) else float(best.baseline_BA23),
                                     "n_tasks": int(best.n_tasks), "tasks_enough": bool(best.tasks_enough)})
    return out
