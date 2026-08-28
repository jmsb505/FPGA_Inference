# Local V4 and 3D Measurement Summary

**Status: completed nine-pair local inference-only run.**

Quality remains sourced from the V4/3D comparison notebook. The table below uses the later matched inference-to-warp benchmark: latency is unmonitored, power is sampled in a separate repeated window of at least 10 seconds per pair, and dynamic energy uses one shared 20-second idle calibration.

| Configuration | Latency (ms) | Mean monitored power (W) | Absolute energy (J) | Dynamic energy (J) |
| --- | ---: | ---: | ---: | ---: |
| 2.5D CPU | 1021.50 +/- 42.68 | 128.73 +/- 7.83 | 131.44 +/- 9.49 | 63.30 +/- 8.50 |
| 2.5D GPU | 273.31 +/- 17.77 | 122.81 +/- 3.75 | 34.10 +/- 1.68 | 15.57 +/- 1.24 |
| 3D CPU | 656.02 +/- 33.30 | 107.92 +/- 6.68 | 71.15 +/- 6.57 | 27.18 +/- 5.06 |
| 3D GPU | 114.73 +/- 11.42 | 185.87 +/- 9.32 | 21.22 +/- 1.52 | 13.57 +/- 0.91 |

Scope: normalized raw in-memory moving/fixed volumes enter preparation; the timer stops when the warped intensity volume is available. Disk I/O, segmentation warp, Dice, TRE, MI, SSIM, plots, and reports are excluded.

Sensor coverage: AMD CPU PPT/package power from LibreHardwareMonitor and NVIDIA GPU board power from `nvidia-smi`. The total excludes RAM, motherboard, storage, display, and PSU losses.

The local result does not complete the cross-platform energy comparison. Run the latest FPGA V4 notebook to produce `inference_pipeline_power_latency_v4.json` with matched inference-only DPU and ARM measurements.
