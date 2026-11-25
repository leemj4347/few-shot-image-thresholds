from __future__ import annotations

import numpy as np
import pandas as pd
import experiment as C

ALPHAS = (0.01, 0.05)
SIZES = (59, 100, 150, 200, 250, 324)
MODE = "base_set"
N = 100
SUBSET_SEED = 20261016


def summarize(runs):
    return runs.groupby(["alpha", "L"]).agg(
        runs=("V", "size"),
        Lambda_mean=("Lambda", "mean"),
        V_mean=("V", "mean"),
        V_p975=("V", lambda values: np.percentile(values, 97.5)),
        TPR_mean=("TPR23", "mean"),
        passing_mean=("passing", "mean"),
        far_ok_mean=("far_ok", "mean"),
        share_runs_V_le_delta=("V", lambda values: float(np.mean(values <= C.DELTA))),
    ).reset_index()


def main(args, out):
    seeds = C.SEEDS[:1] if args.smoke else C.SEEDS
    repeats = 3 if args.smoke else args.repeats
    S, _, _, _, calibration, _, test = C.load_reference()
    names = S.CR.variant_names()
    grid = C.grid()
    C.prepare_test(test, S.SIM.grid_ceil(0.63075), names)
    registries = [(bundle, kind, data) for bundle in test for kind, data in bundle.reg.items()]
    rng = np.random.default_rng(SUBSET_SEED)
    records = []
    for alpha in ALPHAS:
        for seed in seeds:
            model, _, _ = C.load_model(S, N, MODE, alpha, seed)
            cal_rows = [(C.predict(S, model, bundle.reg["base"], alpha, MODE, N),
                         bundle.reg["base"]["Z_eval"]) for bundle in calibration]
            arm = pd.read_parquet(C.arm_path(N, round(1 - alpha, 2), MODE, seed, "np_global"))
            lookup = dict(zip(arm[C.TASK_KEY].itertuples(index=False, name=None), arm.q_pred))
            predictions = [lookup[(bundle.domain, bundle.structure, kind, int(bundle.M), int(bundle.seed))]
                           for bundle, kind, _ in registries]
            for size in SIZES:
                for repeat in range(1 if size == 324 else repeats):
                    indices = (np.arange(len(cal_rows)) if size == 324
                               else np.sort(rng.choice(len(cal_rows), size, replace=False)))
                    lam = C.residual_lambda(S, [cal_rows[i] for i in indices], alpha)["Lambda"]
                    rows = []
                    for (bundle, kind, data), prediction in zip(registries, predictions):
                        tau = C.succ(S.FC.corrected_boundary(prediction, lam), grid)
                        rows.append(C.row(bundle, kind, data, tau, N, alpha, f"L={size}", names, seed=seed))
                    conditions = C.condition_table(pd.DataFrame(rows))
                    records.append(dict(
                        alpha=alpha, training_seed=seed, L=size, repeat=repeat, Lambda=lam,
                        V=conditions.V.mean(), FAR=conditions.FAR.mean(),
                        TPR23=conditions.TPR23.mean(), BA23=conditions.BA23.mean(),
                        passing=int(conditions.meets.sum()), far_ok=int(conditions.meets_far.sum()),
                        sentinel_rate=conditions.sentinel_rate.mean(),
                    ))
    runs = pd.DataFrame(records)
    C.save_csv(runs, out / "calsize_runs.csv")
    C.save_csv(summarize(runs), out / "calsize_summary.csv")


if __name__ == "__main__":
    C.run_step("calibration_size", "Evaluate calibration task counts", main)