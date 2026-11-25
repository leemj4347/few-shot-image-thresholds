# Predictor input ablations and calibration comparisons
from __future__ import annotations

import numpy as np
import pandas as pd
import experiment as C
from experiments import train_evaluate as F2

NS = (10, 20, 50, 100)
BETAS = (0.99, 0.95, 0.80)
CAL = ("fisch_global", "np_global")
ARMS = (
    ("base_set", "raw15", "Meta set w/o standardization", 198401),
    ("set_plain", "normalized15", "Meta set w/o log M", 198145),
    ("set_plain", "raw15", "Fisch DeepSet (no log M, no standardization)", 198145),
)
STATE = {"N": 100}


def encode():
    def enc(d, alpha, mode, domain):
        n = STATE["N"]
        z = np.asarray(d["Z_sup"], dtype=float)[:n]
        if len(z) != n or not np.isfinite(z).all():
            raise ValueError(f"Support set needs {n} finite scores")
        if mode == "set_plain":
            return z[:, None]
        if mode == "base_set":
            return np.column_stack((z, np.full(n, np.log10(d["M"]))))
        raise ValueError(f"Unknown ablation mode: {mode}")
    return enc


def combined_table(summary, main_summary=None, support_summary=None):
    parts = [summary]
    if main_summary is not None:
        methods = ["Meta set + np_global", "Meta set + fisch_global", "Summary H256 + np_global"]
        parts.append(main_summary[main_summary.method.isin(methods)])
    if support_summary is not None:
        parts.append(support_summary[support_summary.method == "Z-norm + NP rank"])
    table = pd.concat(parts, ignore_index=True)
    methods = [
        "Meta set + np_global",
        "Meta set w/o standardization + np_global",
        "Meta set w/o log M + np_global",
        "Fisch DeepSet (no log M, no standardization) + np_global",
        "Summary H256 + np_global",
        "Z-norm + NP rank",
        "Meta set + fisch_global",
        "Meta set w/o standardization + fisch_global",
        "Meta set w/o log M + fisch_global",
        "Fisch DeepSet (no log M, no standardization) + fisch_global",
    ]
    table["order"] = table.method.map(dict(zip(methods, range(len(methods)))))
    table = table[table.alpha.round(2).isin([0.01, 0.05, 0.20])]
    return table.sort_values(["alpha", "n", "order"]).drop(columns="order")


def main(args, out):
    S = C.runner()
    S.encode = encode()
    ns = (100,) if args.smoke else NS
    betas = (0.95,) if args.smoke else BETAS
    definitions = ARMS[2:] if args.smoke else ARMS
    seeds = C.SEEDS[:1] if args.smoke else C.SEEDS
    arms = out / "arms"
    model_root = C.OUT_ROOT / ("models_smoke/ablation" if args.smoke else "models_ablation")
    fit, cal, dev, test, _ = F2.prepare(S, None, out)
    files = []
    for n in ns:
        STATE["N"] = n
        S.CACHE = model_root / f"n{n}"
        for beta in betas:
            alpha = round(1 - beta, 2)
            tag = f"b{int(round(beta * 100)):02d}"
            for mode, variant, label, parameter_count in definitions:
                for seed in seeds:
                    token = f"n{n}_{tag}_{mode}__{variant}_s{seed}"
                    obj, metadata, _ = S.model(mode, variant, alpha, seed, fit, dev)
                    if (metadata["epochs"] != 15 or metadata["pool"] != "mean"
                            or metadata.get("reused") or metadata["parameter_count"] != parameter_count):
                        raise ValueError(f"Unexpected model settings: {token}")
                    for calibration in CAL:
                        correction = S.calibration(obj, mode, alpha, cal, calibration)
                        lam = correction["global"]["Lambda"]
                        rows = []
                        for bundle in test:
                            for kind, registry in bundle.reg.items():
                                q = S.predict(obj, registry, alpha, mode, bundle.domain)
                                tau = S.SIM.strict_successor(S.FC.corrected_boundary(q, lam))
                                rows.append(F2.score(S, bundle, kind, registry, tau, n, alpha,
                                                     f"{label} + {calibration}", seed, q, lam))
                        frame = pd.DataFrame(rows)
                        if len(frame) != 1016:
                            raise ValueError("Expected 1016 registries")
                        arm = f"{token}_{calibration}"
                        F2.save_arm(frame, arm, arms)
                        files.append(arms / f"{arm}.conditions.csv")
    conditions = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    C.save_csv(conditions, out / "conditions.csv")
    summary = C.method_summary(conditions)
    C.save_csv(summary, out / "summary.csv")
    main_path = C.stored(C.RESULTS / "main/summary.csv")
    support_path = C.stored(C.RESULTS / "support_baselines/summary.csv")
    main_summary = pd.read_csv(main_path) if main_path.exists() else None
    support_summary = pd.read_csv(support_path) if support_path.exists() else None
    C.save_csv(combined_table(summary, main_summary, support_summary), out / "ablation_table.csv")


if __name__ == "__main__":
    C.run_step("ABL", "Train and evaluate predictor ablations", main)