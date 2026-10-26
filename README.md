# Few-shot Image Registry Thresholds

Support-based threshold prediction and residual calibration for image-copy detection.

## Requirements

Python 3.12, PyTorch and an NVIDIA GPU for SSCD. Package versions are in `requirements.txt`; execution scripts use PowerShell.

## Installation

```powershell
./setup_env.ps1
```

## Data

Dataset locations are configured in `configs/paths.json` (see `configs/paths.example.json`).
Reference input files and checksums are listed in `configs/inputs.json`; `prepare_inputs.py` stages these files. Reference inputs are not bundled, and a public download is not yet available.

## Running experiments

```powershell
./run_all.ps1
```

Run an individual stage or analyze stored results:

```powershell
./run_step.ps1 -Step main
./run_analysis.ps1
```

Metrics are saved in `results/`, and additional analyses in `runs/`.

## External resources

- [Fisch et al. source](https://github.com/ajfisch/few-shot-cp/tree/bcd18d1726541da827f7530a724460bd735bc0ed) → `external/fisch/`. File checksums are in `configs/dependencies.json`.
- [SSCD weights](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt) → `weights/sscd_disc_mixup.torchscript.pt` (also downloaded by the `sscd_download` stage).
- [DISC21 images and ground truth](https://ai.meta.com/datasets/disc21-dataset/).
