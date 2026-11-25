from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment as C
from experiments.s16_ablation import combined_table


def main():
    parser = argparse.ArgumentParser(description="Recalculate predictor ablation results.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--main-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.data / "arms").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(args.data / "arms/*.parquet")
    args.output.mkdir(parents=True, exist_ok=False)
    conditions = pd.concat([C.condition_table(pd.read_parquet(p)) for p in files], ignore_index=True)
    summary = C.method_summary(conditions)
    C.save_csv(conditions, args.output / "conditions.csv")
    C.save_csv(summary, args.output / "summary.csv")
    main_summary = support_summary = None
    if args.main_data is not None:
        main_summary = pd.read_csv(args.main_data / "main/summary.csv")
        support_rows = pd.read_parquet(args.main_data / "support_baselines/rows_new_baselines.parquet")
        support_summary = C.method_summary(C.condition_table(support_rows))
    C.save_csv(combined_table(summary, main_summary, support_summary), args.output / "ablation_table.csv")


if __name__ == "__main__":
    main()