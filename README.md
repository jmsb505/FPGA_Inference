# V4 2.5D FPGA registration

This repository contains the V4 2.5D medical-image registration workflow used to compare local CPU/GPU execution with FPGA ARM CPU and DPU deployment. The reported evaluation uses mean fusion across three 2D orientation flows and a 3D GPU baseline.

## Active workflow

Run commands from the repository root. The local Python environment is described in `environment.yml`; the Vitis AI quantization and compilation steps require the matching Vitis AI 2.5 container and board image.

```bash
conda env create -f environment.yml
conda activate gen
python Voxelmorph/2.5D/train_2p5d_pytorchV4.py --build-cache
python Voxelmorph/2.5D/prepare_calibration_data_v4.py
python vitis_run_v4.py
python windows_metrics_package/run_windows_benchmark.py --check-only
python windows_metrics_package/run_windows_benchmark.py
```

The V4 FPGA input contract is NHWC `[1, 112, 96, 16]`: eight moving slices followed by eight fixed slices. The orientation-aware letterbox/transposition contract must be preserved. Images use trilinear resampling and segmentation labels use nearest-neighbor resampling.

## Repository policy

Git contains source code, compact result tables, and documentation. Private volumes, calibration tensors, checkpoints, compiled FPGA artifacts, rendered figures, and transfer archives are intentionally excluded. Use `--data-root` or `REGISTRATION_DATA_ROOT` where supported to point tools at the local dataset.

See `PROJECT_EXPERIMENT_REPORT.md` for the project narrative and `DOCUMENTATION_INDEX.md` for the current result sources.
