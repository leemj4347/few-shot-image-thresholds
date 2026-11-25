from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment as C
from experiments.calibration_size import summarize


def main():
    parser = argparse.ArgumentParser(description="Recalculate calibration task-count results.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = pd.read_csv(args.data)
    summary = summarize(runs)
    args.output.mkdir(parents=True, exist_ok=False)
    C.save_csv(summary, args.output / "calsize_summary.csv")


if __name__ == "__main__":
    main()