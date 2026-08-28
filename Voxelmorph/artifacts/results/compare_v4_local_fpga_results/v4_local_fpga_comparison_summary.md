# V4 Local vs FPGA Comparison

**Status (2026-08-26): quality and matched local/FPGA inference-to-warp measurements complete.**

## Quality

Nine aligned private-dataset pairs use the V4 canonical-letterbox pipeline and segmentation-derived label-centroid TRE.

| Configuration | Method | Dice | TRE (mm) | MI | SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| FPGA DPU | Mean fused | 0.7647 | 2.025 | 0.5957 | 0.7415 |
| FPGA ARM CPU | Mean fused | 0.7638 | 2.008 | 0.5942 | 0.7416 |
| Local CPU/GPU | Mean fused | 0.7638 | 2.008 | 0.5942 | 0.7416 |
| Local 3D V2 | Single model | 0.7609 | about 2.111 | 0.5772 | 0.7459 |

The DPU retains ARM-reference quality closely. Mean fusion is preferred over axial or smoothed V4. The 3D model is local-only.

## Local inference-only measurement

This window starts with normalized in-memory volumes and ends with the warped intensity volume. It excludes segmentation warping and quality metrics.

| Configuration | Latency (ms) | Mean power (W) | Absolute energy (J) | Dynamic energy (J) |
| --- | ---: | ---: | ---: | ---: |
| 2.5D CPU | 1021.50 | 128.73 | 131.44 | 63.30 |
| 2.5D GPU | 273.31 | 122.81 | 34.10 | 15.57 |
| 3D CPU | 656.02 | 107.92 | 71.15 | 27.18 |
| 3D GPU | 114.73 | 185.87 | 21.22 | 13.57 |

## FPGA matched inference-to-warp measurement

| Configuration | Accelerator/model (ms) | ARM host/non-model (ms) | Total latency (ms) | Mean power (W) | Absolute energy (J) | Dynamic energy (J) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5D DPU | 697.86 | 12390.01 | 13087.88 | 1.8475 | 24.243 | 2.495 |
| 2.5D ARM CPU | 28103.29 | 10155.64 | 38258.93 | 2.0941 | 81.136 | 16.921 |

The complete-pipeline warm-up removed the earlier first-pair timing outlier. DPU inference-to-warp is 2.92 times faster than ARM CPU and uses 70.1% less absolute monitored-rail energy and 85.3% less idle-subtracted dynamic energy. Host work still accounts for 94.7% of DPU-path latency.

## Energy guardrails

The inference-only values above are directly sampled, not estimated from the older full-evaluation run. Local inference-only totals use CPU package/PPT plus NVIDIA board telemetry. FPGA values use PSINTFP and INT rails. Neither is whole-system wall power, so the strongest energy comparisons remain within each platform.
