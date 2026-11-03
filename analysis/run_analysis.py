from pathlib import Path
import argparse
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--data')
    parser.add_argument('--experiment-python', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.data:
        subprocess.run([args.experiment_python, CODE / 'recalculate.py', '--data', args.data, '--output', output], cwd=ROOT, check=True)
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'supplement').mkdir()
    env = os.environ.copy()
    env.update(ANALYSIS_OUTPUT=str(output / 'supplement'), EXPERIMENT_RESULTS=str(ROOT / 'results'),
               PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('RUN_ROOT', None)
    stages = [
        ('bootstrap_analysis.py', ['conditional_intervals.csv', 'paired_differences.csv', 'point_estimates.csv', 'task_flow.csv']),
        ('operating_points.py', ['development_adjusted_rows.parquet', 'development_adjusted_summary.csv', 'development_selection.csv']),
        ('operating_point_intervals.py', ['development_fixed_paired_differences.csv', 'development_fixed_intervals.csv'])]
    for script, required in stages:
        subprocess.run([args.experiment_python, CODE / script], cwd=ROOT, env=env, check=True)
        missing = [name for name in required if not (output / 'supplement' / name).is_file()]
        if missing:
            raise RuntimeError(f'{script}: missing outputs {missing}')

if __name__ == '__main__':
    main()
