# V4 Registration Results: FPGA Board and Local WSL Analysis

**Status (2026-08-26): quality, matched local inference-only, and matched FPGA inference-only measurements complete.**

## Experiment Scope

- Nine deterministic held-out private-dataset MR-to-CT pairs.
- V4 2.5D uses one canonical-letterbox model for axial, coronal, and sagittal inference.
- DPU input is fixed NHWC `[1,112,96,16]`.
- 3D V2 is local-only and was never deployed on FPGA.
- TRE is segmentation-derived anatomical-label-centroid distance in millimetres, not manual-landmark TRE.

### Source files

| Scope | Source |
| --- | --- |
| Board/local quality and original full-evaluation telemetry | `Voxelmorph/benchmark_results_comparison_v4.json` and `benchmark_results_comparison_v4_local.json` |
| Local inference-only latency and power | `Voxelmorph/artifacts/results/inference_only_local_power_latency/local_2p5d_3d_cpu_gpu_9_pairs.json` |
| Final matched FPGA latency, power, and energy | Voxelmorph/artifacts/results/inference_pipeline_breakdown/inference_pipeline_power_latency_v4.json |
| Local V4/3D quality rows | `Voxelmorph/artifacts/results/compare_2p5d_v4_fusion_3d_v2_test/v4_vs_3d_rows.json` |

## Quality Results

| Configuration | Method | Dice before | Dice after | TRE before (mm) | TRE after (mm) | MI after | SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FPGA DPU | Axial | 0.5712 | 0.7122 | 3.724 | 2.520 | 0.6043 | 0.7402 |
| FPGA DPU | Mean fused | 0.5712 | 0.7647 | 3.724 | 2.025 | 0.5957 | 0.7415 |
| FPGA DPU | Smoothed | 0.5712 | 0.7599 | 3.724 | 2.036 | 0.5995 | 0.7398 |
| FPGA ARM CPU | Axial | 0.5712 | 0.7160 | 3.724 | 2.500 | 0.6063 | 0.7405 |
| FPGA ARM CPU | Mean fused | 0.5712 | 0.7638 | 3.724 | 2.008 | 0.5942 | 0.7416 |
| FPGA ARM CPU | Smoothed | 0.5712 | 0.7592 | 3.724 | 2.014 | 0.5988 | 0.7401 |
| Local CPU/GPU V4 | Mean fused | 0.5712 | 0.7638 | 3.724 | 2.008 | 0.5942 | 0.7416 |
| Local 3D V2 | Single model | 0.5712 | 0.7609 | 3.724 | about 2.111 | 0.5772 | 0.7459 |

Mean fusion is the recommended V4 method. On the DPU it improves Dice by 0.0525 and reduces TRE by 0.495 mm relative to axial inference. Smoothing does not improve Dice or TRE. Quantized DPU mean fusion differs from the ARM reference by +0.0009 Dice and +0.017 mm TRE, which supports the conclusion that V4 quantization preserved quality on this cohort.

## Measurement Scopes

### Refreshed full quality-evaluation scope

The board/local reports wrap registration and quality work together. The following board values include long host-side processing and quality evaluation:

| Configuration | Method | Model (ms) | Recorded total (ms) | Absolute energy (J) | Dynamic energy (J) |
| --- | --- | ---: | ---: | ---: | ---: |
| FPGA DPU | Mean fused | 1822.5 | 99198.0 | 186.528 | 16.626 |
| FPGA ARM CPU | Mean fused | 29472.1 | 126766.7 | 241.191 | 27.055 |

Within this scope, DPU model execution is 16.17 times faster and recorded dynamic energy is 38.6% lower than ARM CPU. The table does not measure inference-only energy. It must not be compared directly with the newer local inference-to-warp table.

### Corrected local inference-only scope

Start: normalized raw in-memory moving/fixed volumes enter model preparation.

End: the warped moving intensity volume is available in host memory.

Included: preparation, transfers, model execution, three-orientation fusion for 2.5D or 3D flow handling, resizing, and intensity warping.

Excluded: disk I/O, segmentation warp, Dice, TRE, MI, SSIM, plotting, and reports.

Latency is measured without monitoring. Power uses a separate repeated execution lasting at least 10 seconds per pair and one shared 20-second idle calibration.

| Configuration | Latency (ms) | Mean power (W) | Absolute energy (J) | Dynamic energy (J) |
| --- | ---: | ---: | ---: | ---: |
| 2.5D CPU | 1021.50 +/- 42.68 | 128.73 +/- 7.83 | 131.44 +/- 9.49 | 63.30 +/- 8.50 |
| 2.5D GPU | 273.31 +/- 17.77 | 122.81 +/- 3.75 | 34.10 +/- 1.68 | 15.57 +/- 1.24 |
| 3D CPU | 656.02 +/- 33.30 | 107.92 +/- 6.68 | 71.15 +/- 6.57 | 27.18 +/- 5.06 |
| 3D GPU | 114.73 +/- 11.42 | 185.87 +/- 9.32 | 21.22 +/- 1.52 | 13.57 +/- 0.91 |

Local conclusions:

- 3D GPU is 2.38 times faster than 2.5D GPU for the complete inference-to-warp window.
- 3D GPU draws more mean monitored power but uses less absolute and dynamic energy because its duration is shorter.
- 2.5D GPU is 3.74 times faster than 2.5D CPU and reduces absolute energy by about 74%.
- These totals combine Ryzen CPU PPT/package and NVIDIA board power. They exclude RAM, motherboard, storage, display, and PSU losses.

### Matched FPGA inference-to-warp latency, power, and energy

The final board benchmark uses the same inference-to-warp boundary as the local run. Latency is collected without monitoring over three repetitions. A separate repeated execution supplies at least 10 seconds of 0.1-second rail samples after a shared 20-second idle calibration.

| Configuration | Accelerator/model (ms) | ARM host/non-model (ms) | Total latency (ms) | Mean power (W) | Absolute energy (J) | Dynamic energy (J) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5D DPU | 697.86 | 12390.01 | 13087.88 +/- 74.83 | 1.8475 +/- 0.0109 | 24.243 +/- 0.103 | 2.495 +/- 0.138 |
| 2.5D ARM CPU | 28103.29 | 10155.64 | 38258.93 +/- 213.38 | 2.0941 +/- 0.0094 | 81.136 +/- 0.372 | 16.921 +/- 0.257 |

The complete warm-up removed the earlier 27.29-second first-pair outlier: DPU total latency now has a 0.075-second between-pair standard deviation. The DPU path is 2.92 times faster than ARM CPU, with 70.1% lower absolute monitored-rail energy and 85.3% lower idle-subtracted dynamic energy. However, the DPU kernel is only 5.3% of its total path; ARM host preparation, lifting, fusion, resizing, and warping account for 94.7%.

PSINTFP plus INT is a board-rail boundary, while desktop totals combine CPU package/PPT and NVIDIA board telemetry. Neither is wall power, so direct cross-platform energy rankings remain descriptive rather than physically equivalent.
## Errors and Corrective Work

| Error | Effect | Correction |
| --- | --- | --- |
| Variable orientation sizes | No single DPU graph | Canonical `[96,112,96]` grid and fixed `112x96x16` interface |
| Direct stretching considered | Possible aspect-ratio distortion | Orientation-aware padding/transposition |
| Pre-V4 fused FPGA result below axial | Multi-view evidence was reconstructed inconsistently | Shared V4 geometry, inverse mapping, and calibration contract |
| WSL eager loading crash | Training initialization failed | Lazy loading, persistent canonical cache, bounded subject cache |
| Very slow training estimate | Full run was impractical | Cache reuse, larger subject cache, resumable checkpoints |
| Missing board `.npy` files | All nine pairs skipped | Upload complete 18-subject test bundle |
| Empty aggregation after skipped pairs | `np.max` raised on an empty list | Guard aggregation and fail with a useful missing-data error |
| No manual landmarks | Conventional TRE unavailable | Segmentation-centroid TRE with explicit limitations |
| LibreHardwareMonitor unreachable from WSL | CPU power was null | Enable web server and use WSL Windows-host gateway discovery |
| Metrics inside latency/power window | Inference time and energy were overstated | Separate quality, unmonitored latency, and monitored inference-only passes |
| One-slice board warm-up | First DPU pair was an outlier | Warm the complete inference-to-warp path |

## Interpretation and Conclusions

- V4 fixed geometry resolved the earlier FPGA fusion inconsistency.
- Mean fusion is the best V4 quality choice.
- DPU quantization preserved mean-fused quality relative to ARM float inference.
- The current board implementation is host-bound even after metrics are removed.
- The local 3D model is faster and lower-energy on the RTX 4070 Ti, but this does not establish deployability on the ZCU104.
- Board and desktop watts cover different rails. Cross-platform absolute power rankings are descriptive, not physically equivalent.
- The older 38.6% DPU energy reduction remains valid only within the board full-evaluation scope; the matched inference-only run shows 85.3% lower dynamic energy and 70.1% lower absolute energy than ARM CPU.
- Direct inference-only FPGA energy remains pending.

## Final Measurement Validation

The final file contains schema version 2, pair_count 9, nine DPU rows, nine ARM rows, pair indices 0-8, and subject identities matching the earlier timing benchmark. All latency, power, energy, and idle-baseline aggregates are finite. The figures and summaries now read this file directly; no FPGA inference-energy estimate is used.

Future reruns must keep quality evaluation separate from inference-to-warp measurement and must preserve the same warm-up, idle calibration, sensor rails, and pair ordering.
