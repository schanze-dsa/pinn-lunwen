# pinn-lunwen

This repository is a trimmed but runnable export of the current PINN mainline experiment. After cloning, you can create a Python environment, install dependencies, and launch the same 48h-budget two-stage training configuration used for the paper-oriented mainline run.

## Included runtime scope

- core training entry: `main_new.py`, root `config.yaml`
- required source packages: `assembly/`, `inp_io/`, `mesh/`, `model/`, `physics/`, `train/`, `viz/`
- mainline configs: `configs/config.yaml`, `configs/paper_mainline_best.yaml`
- current runtime data: `mir111.cdb`, `ansys_cases_180_deg2to6_step0p5_pinn.csv`, `rigid_removed_csv/`
- launch scripts: `scripts/setup_env.ps1`, `scripts/run_paper_mainline_best.ps1`
- optical bridge scripts: `scripts/export_zemax_grid_sag.py`, `scripts/run_zemax_energy_concentration.ps1`, `scripts/run_zemax_optical_eval.ps1`

## Intentionally excluded

- `.git`, virtual environments, `__pycache__`, logs, checkpoints, results, tmp/output folders
- tests, exploratory notebooks/scripts, local search logs, and machine-specific artifacts
- resume checkpoints and absolute local filesystem paths

## Environment

- Python: `3.8.x`
- Verified package stack:
  - `tensorflow==2.10.1`
  - `numpy==1.23.5`
  - `pandas==2.0.3`
  - `scipy==1.5.4`
  - `matplotlib==3.7.5`
  - `tqdm==4.66.5`
  - `PyYAML==6.0.2`

## Quick start

```powershell
git clone https://github.com/schanze-dsa/pinn-lunwen.git
cd pinn-lunwen
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Python python
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_paper_mainline_best.ps1 -Python .\.venv\Scripts\python.exe -Config .\configs\paper_mainline_best.yaml
```

## Training configuration

- default mainline route: normal-contact-first
- annular modal residual branch: enabled
- stage delta consistency: `w_delta_data = 0.20`
- two-stage budget:
  - `phase1.max_steps = 18000`
  - `phase2.max_steps = 9000`

All checkpoint resume fields have been reset to fresh-run values, so the cloned repository starts from scratch.

## Optical evaluation

The Zemax bridge is optional and requires a local OpticStudio installation with a valid ZOS-API license.

```powershell
python .\scripts\export_zemax_grid_sag.py --help
powershell -ExecutionPolicy Bypass -File .\scripts\run_zemax_energy_concentration.ps1 -LensFile C:\path\to\system.zmx
```
