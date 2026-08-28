# V4 Local Metrics Package (WSL)

Run this package from the same WSL Python environment as the V4 notebook. It runs `Voxelmorph/compare_2p5d_v4_fusion_3d_v2_test.ipynb` and writes quality, timing, and board-parity power fields.

## Measurement Protocol

After a V4 warm-up, the notebook collects a 3-second idle baseline. It measures axial, coronal, sagittal, and each post-processing stage separately, using the same FPGA formula:

```text
energy_j = mean_power_w * measured_duration_s
dynamic_energy_j = max(energy_j - idle_power_w * measured_duration_s, 0)
```

Short CUDA stages repeat 25 times for stable sampling. Their energy and timing are normalized per logical registration run; quality is evaluated once.

## WSL Power Backends

- GPU: NVML (`nvidia-ml-py`) in WSL, with `nvidia-smi` fallback.
- CPU: Linux RAPL if WSL exposes `/sys/class/powercap/intel-rapl:*/energy_uj`.
- CPU fallback: LibreHardwareMonitor on Windows with `Options > Remote Web Server > Run`; modern WSL accesses its `http://localhost:8085/data.json` endpoint.

If neither CPU source is available, CPU and total-energy fields remain `null`; GPU energy remains valid.

## Setup and Sensor Check

In WSL, activate the environment used for the V4 notebook and install the supplemental packages. Windows-only WMI dependencies are skipped automatically.

```bash
conda activate gen
pip install -r windows_metrics_package/requirements_windows_metrics.txt
python windows_metrics_package/run_windows_benchmark.py --check-only
```

The check writes `Voxelmorph/artifacts/results/compare_2p5d_v4_fusion_3d_v2_test/windows_sensor_check.json`. Confirm `gpu_power_w` is numeric. For CPU power, inspect `linux_rapl_energy_paths` or enable the LibreHardwareMonitor web server on Windows.

## Run

Start with a one-pair smoke run:

```bash
python windows_metrics_package/run_windows_benchmark.py --pair-limit 1
```

Run every held-out pair:

```bash
python windows_metrics_package/run_windows_benchmark.py
```

Adjust only the measurement repetitions when needed:

```bash
python windows_metrics_package/run_windows_benchmark.py \
  --pair-limit 1 --power-inference-repetitions 25 --power-postprocess-repetitions 1
```

Outputs are written to `Voxelmorph/artifacts/results/compare_2p5d_v4_fusion_3d_v2_test/`, including `benchmark_results_comparison_v4_local.json` and `windows_metrics_report_v4.json`. Compare dynamic energy within a platform; desktop and FPGA raw joules are not directly interchangeable.
