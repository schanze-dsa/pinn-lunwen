# pinn-lunwen

This repository is a minimal control-plane export for the PINN paper workflow. It intentionally contains only the core training configuration and the automation scripts needed to launch the mainline experiment or connect the predicted mirror deformation to the Zemax optical evaluation bridge.

## Included files

- `configs/config.yaml`
- `configs/paper_mainline_best.yaml`
- `scripts/run_paper_mainline_best.ps1`
- `scripts/export_zemax_grid_sag.py`
- `scripts/run_zemax_energy_concentration.ps1`
- `scripts/run_zemax_optical_eval.ps1`

## What is intentionally excluded

- source code modules outside the selected automation scripts
- training data, mesh data, FEM results, checkpoints, logs, and temporary outputs
- local virtual environments, caches, compiled artifacts, and editor metadata
- machine-specific resume checkpoints and absolute local paths

## Notes

- The exported YAML files preserve the current 48h mainline setup, including the annular residual branch and `w_delta_data = 0.20`, but all resume-from-checkpoint fields have been reset to neutral values.
- `scripts/run_paper_mainline_best.ps1` assumes the full training codebase is available alongside these files and invokes `main_new.py --config ...`.
- The Zemax scripts require a local OpticStudio installation with a valid ZOS-API license.

## Typical usage

```powershell
pwsh -File .\scripts\run_paper_mainline_best.ps1 -Python python -Config .\configs\paper_mainline_best.yaml
```

```powershell
python .\scripts\export_zemax_grid_sag.py --help
pwsh -File .\scripts\run_zemax_energy_concentration.ps1 -LensFile C:\path\to\system.zmx
```
