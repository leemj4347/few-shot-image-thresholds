# Few-shot Image Registry Thresholds

Support-based threshold prediction and residual calibration for image-copy detection.

## Numerical results

Download [registry-metrics.zip](https://github.com/leemj4347/few-shot-image-thresholds/releases/download/paper-results/registry-metrics.zip) and extract it.

Python 3.12:

```shell
python -m pip install -r requirements-analysis.txt
python analysis/recalculate.py --data /path/to/metrics --output /path/to/results
```

The output contains condition summaries, registry FAR order statistics, edit-wise detection rates and paired bootstrap intervals. Rates are proportions; `schema.json` describes the inputs. Development-selected shifts remain fixed during interval calculation.

The archive contains evaluation measurements. It supports recalculation of the reported numerical analyses, rather than model retraining or image-score generation.

## Predictor ablations

Download [ablation-results.zip](https://github.com/leemj4347/few-shot-image-thresholds/releases/download/predictor-ablations/ablation-results.zip) and extract it. It contains registry-level results for three input variants, both calibration rules, four support sizes, three nominal levels and five seeds; model weights are not included.

```shell
python analysis/recalculate_ablation.py --data /path/to/results/ABL --main-data /path/to/metrics --output /path/to/ablation-summary
```

The `--main-data` directory is the numerical archive above. It supplies the proposed predictor, H256 and Z-norm reference rows for the combined table.

## Calibration task counts

Download [calibration-size-results.zip](https://github.com/leemj4347/few-shot-image-thresholds/releases/download/calibration-size/calibration-size-results.zip) and extract it. Results use 100 support queries, calibration task counts of 59–324, and five training seeds.

```shell
python analysis/recalculate_calibration_size.py --data /path/to/results/calibration_size/calsize_runs.csv --output /path/to/calibration-size-summary
```

## Image experiments

```powershell
./setup_env.ps1
Copy-Item configs/paths.example.json configs/paths.json
python prepare_inputs.py --source /path/to/task-inputs --fisch-source /path/to/few-shot-cp
./run_all.ps1
```

Set the image locations in `configs/paths.json`. The task-input directory must contain the relative files listed in `configs/inputs.json`; these inputs are separate from the numerical archive. Full image experiments require the source image pools, task inputs and PyTorch. SSCD inference requires an NVIDIA GPU with CUDA.

Individual stages and additional analyses:

```powershell
./run_step.ps1 -Step main
./run_step.ps1 -Step ablation
./run_step.ps1 -Step calibration_size
./run_analysis.ps1
```

Use empty result and training-output directories for a new experiment. Completed models remain available to downstream evaluation stages. Numerical results are written to `results/`; additional analyses to `runs/`.

## External resources

- [Few-shot Conformal Prediction with Auxiliary Tasks](https://github.com/ajfisch/few-shot-cp/tree/bcd18d1726541da827f7530a724460bd735bc0ed): obtain the author source at this commit and pass its root to `prepare_inputs.py --fisch-source`. Required files are listed in `configs/dependencies.json`. The source is not redistributed here.
- [SSCD](https://github.com/facebookresearch/sscd-copy-detection): [TorchScript weights](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt), downloaded by `sscd_download`. The upstream code uses the [MIT license](https://github.com/facebookresearch/sscd-copy-detection/blob/main/LICENSE).
- [DISC21](https://ai.meta.com/datasets/disc21-downloads/): obtain images and annotations from the provider under its dataset agreement. Source datasets and their licenses remain with their providers.
