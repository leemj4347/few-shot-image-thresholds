from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment as C


def aggregate(rows, out, by=('method', 'n', 'alpha')):
    out.mkdir(parents=True, exist_ok=False)
    conditions = C.condition_table(rows)
    C.save_csv(conditions, out / 'conditions.csv')
    C.save_csv(C.method_summary(conditions, by=by), out / 'summary.csv')
    return conditions


def main():
    parser = argparse.ArgumentParser(description='Recalculate registry-level numerical results.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    out = C.output_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    C.RESULTS = data
    C.BASE_RESULTS = data
    C.ARMS = data / 'main/arms'
    frames = [pd.read_parquet(p) for p in sorted(C.ARMS.glob('*.parquet'))]
    if not frames:
        raise FileNotFoundError('main/arms/*.parquet')
    rows = pd.concat(frames, ignore_index=True)
    aggregate(rows, out / 'main')
    support = pd.read_parquet(data / 'support_baselines/rows_new_baselines.parquet')
    keep = ['Meta set + np_global', 'Meta set + fisch_global', 'Direct NP', 'Direct split CP', 'DBpHash fixed', 'Youden J']
    old = rows[rows.method.isin(keep) & rows.alpha.isin([0.01, 0.05])]
    hamming = pd.read_parquet(data / 'hash_thresholds/hash_convention_rows.parquet')
    hamming = hamming[hamming.alpha.isin([0.01, 0.05])]
    aggregate(pd.concat([support, old, hamming], ignore_index=True), out / 'support_baselines')
    for name in ['phash', 'sscd', 'disc21']:
        by = ('scheme', 'method', 'n', 'alpha') if name == 'disc21' else ('method', 'n', 'alpha')
        aggregate(pd.read_parquet(data / name / 'rows.parquet'), out / name, by=by)
    coverage = pd.read_csv(data / 'coverage/conditions.csv')
    (out / 'coverage').mkdir()
    C.save_csv(C.method_summary(coverage), out / 'coverage/summary.csv')
    condkeys = ['domain', 'structure', 'kind', 'M', 'n', 'alpha', 'training_seed']
    rank_input = pd.concat([rows, pd.read_parquet(data / 'hash_thresholds/hash_convention_rows.parquet')], ignore_index=True)
    condkeys = ['method'] + condkeys
    rankrows = []
    for key, g in rank_input.groupby(condkeys):
        rank = int(np.floor(C.DELTA * len(g))) + 1
        rankrows.append(dict(zip(condkeys, key), registries=len(g), rank=rank,
                             FAR_order_statistic=float(np.sort(g.FAR)[-rank])))
    C.save_csv(pd.DataFrame(rankrows), out / 'main/registry_far_order_statistics.csv')
    from experiments import edit_detection
    (out / 'edit_detection').mkdir()
    edit_detection.main(SimpleNamespace(smoke=False, log=None), out / 'edit_detection')
    additional = out / 'additional'
    additional.mkdir()
    selected = pd.read_parquet(data / 'additional/development_adjusted_rows.parquet')
    selected.to_parquet(additional / 'development_adjusted_rows.parquet', index=False)
    shutil.copyfile(data / 'additional/development_selection.csv', additional / 'development_selection.csv')
    keys = ['n', 'alpha', 'method', 'training_seed', 'development_constraint', 'domain', 'structure', 'kind', 'M']
    grouped = selected.groupby(keys).agg(V=('violation', 'mean'), FAR=('FAR', 'mean'), TPR23=('TPR23', 'mean'),
                                        BA23=('BA23', 'mean'), baseline_BA23=('baseline_BA23', 'mean'),
                                        sentinel_rate=('sentinel', 'mean')).reset_index()
    grouped['meets'] = (grouped.V <= 0.05 + 1e-12) & (grouped.BA23 >= grouped.baseline_BA23 - 1e-12)
    C.save_csv(grouped, additional / 'development_adjusted_conditions.csv')
    byseed = grouped.groupby(['n', 'alpha', 'method', 'training_seed', 'development_constraint']).agg(
        V=('V', 'mean'), FAR=('FAR', 'mean'), TPR23=('TPR23', 'mean'), BA23=('BA23', 'mean'), passing=('meets', 'sum')).reset_index()
    C.save_csv(byseed, additional / 'development_adjusted_by_seed.csv')
    summary = byseed.groupby(['n', 'alpha', 'method', 'development_constraint'])[['V', 'FAR', 'TPR23', 'BA23', 'passing']].mean().reset_index()
    C.save_csv(summary, additional / 'development_adjusted_summary.csv')
    env = os.environ.copy()
    env.update(EXPERIMENT_RESULTS=str(data), ANALYSIS_OUTPUT=str(additional), PYTHONDONTWRITEBYTECODE='1')
    env.pop('RUN_ROOT', None)
    for script in ['bootstrap_analysis.py', 'operating_point_intervals.py']:
        subprocess.run([sys.executable, Path(__file__).parent / script], cwd=ROOT, env=env, check=True)
    from experiments import statistical_analysis
    (out / 'statistics').mkdir()
    statistical_analysis.main(SimpleNamespace(smoke=False, bootstrap=2000, log=None), out / 'statistics')
    from scipy.stats import ks_2samp
    cal = pd.read_parquet(data / 'residuals/residuals_calibration.parquet')
    test = pd.read_parquet(data / 'residuals/residuals_test.parquet')
    residuals = []
    for (alpha, domain, kind), group in test.groupby(['alpha', 'domain', 'kind']):
        quantiles = np.percentile(group.residual, [5, 50, 95])
        residuals.append(dict(alpha=alpha, domain=domain, kind=kind, registries_x_seeds=len(group),
            residual_p05=quantiles[0], residual_median=quantiles[1], residual_p95=quantiles[2],
            share_exceeds_lambda=float(group.exceeds_lambda.mean()), violation_rate=float(group.violation.mean()),
            ks_stat_vs_calibration=float(ks_2samp(group.residual, cal[cal.alpha == alpha].residual).statistic)))
    C.save_csv(pd.DataFrame(residuals), out / 'residual_by_domain_kind.csv')


if __name__ == '__main__':
    main()
