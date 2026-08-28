# Project Report: Fixed-Grid 2.5D VoxelMorph for FPGA Deployment

**Documentation status: 2026-08-26.** Quality and matched inference-to-warp latency, power, and energy measurements are complete for the nine recorded pairs. The final FPGA result file is validated and stored in the results tree.

## Executive Summary

This project evaluates deformable MR-to-CT registration with VoxelMorph-style networks and deploys a quantized 2.5D model on an AMD/Xilinx ZCU104 DPU. The work progressed from 2D and 3D local experiments through several 2.5D versions, Vitis AI export, fixed-grid V4 training, board deployment, quality evaluation, and hardware measurement.

V4 mean fusion is the strongest measured FPGA quality configuration. Across nine held-out private-dataset pairs, the quantized DPU achieved Dice 0.7647 and segmentation-derived anatomical-label-centroid TRE 2.025 mm. The FPGA ARM floating-point reference achieved Dice 0.7638 and TRE 2.008 mm. These results show that the V4 fixed geometry and INT8 compilation preserve registration quality closely.

Three measurement scopes now exist and must not be mixed:

1. **Quality evaluation:** registration, segmentation warp, Dice, TRE, MI, and SSIM.
2. **Inference-to-warp latency:** normalized in-memory volumes enter model preparation and the warped intensity volume is returned.
3. **Inference-only power:** a separate execution of the same inference-to-warp path, excluding quality metrics and report generation.

The matched inference-only experiment is complete for local 2.5D/3D CPU/GPU and board 2.5D DPU/ARM CPU. Earlier board energy tables cover the full quality-evaluation pipeline and remain historical evidence for that scope only.

## 1. Research Goal

The deployment question was whether a learned deformable registration model could retain useful anatomical alignment while satisfying a static FPGA tensor contract. The final system needed to:

- use one compiled model across axial, coronal, and sagittal views;
- make training, calibration, local inference, and board inference geometrically consistent;
- retain quality after INT8 quantization;
- compare 2.5D deployment against a local 3D reference;
- report quality, latency, power, and energy without crossing incompatible measurement boundaries.

The 3D model was kept as a local baseline. It was never compiled or tested on FPGA, so no 3D FPGA claim is made.

## 2. Dataset and Metrics

The active workflow uses centered private-dataset MR and CT volumes and matching centered segmentation maps. The evaluation cohort contains 18 subjects arranged into nine deterministic cross-subject MR-to-CT pairs. Recorded metadata indicate matching volume/segmentation geometry and 1 mm isotropic spacing for the evaluated data.

Sixteen segmentation labels are used: `2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28`.

Metrics have different roles:

- **Dice** measures anatomical-label overlap.
- **Mutual information (MI)** measures multimodal intensity dependence.
- **SSIM** measures structural image similarity.
- **TRE** is the mean distance in millimetres between corresponding warped-moving and fixed label centroids.

The TRE implementation is segmentation-derived anatomical-label-centroid TRE. It is not manual-landmark TRE. It supplies a physical-distance measure, but it can hide local boundary errors and depends on segmentation quality. Dice and TRE must therefore be interpreted together.

## 3. Model Evolution and Design Decisions

| Decision | Alternatives | Evidence and justification |
| --- | --- | --- |
| Deploy 2.5D | 2D or full 3D FPGA deployment | 2.5D adds through-plane context while keeping the network and activation footprint far below the local 3D model. The 3D reference reached about 1,746 MB CUDA peak allocation versus about 110 MB for 2.5D in the earlier resource benchmark. |
| One model for three orientations | Three separately trained models | One graph reduces training, quantization, packaging, and maintenance work. V4 shows that one shared model can benefit from mean fusion when geometry is consistent. |
| Fixed NHWC `[1,112,96,16]` | Variable sizes or several compiled graphs | Vitis AI requires a static graph. Sixteen channels contain eight moving slices followed by eight fixed slices. |
| Canonical volume `[96,112,96]` | Native subject shapes | A canonical grid makes extraction, lifting, fusion, and inverse mapping deterministic across subjects and platforms. |
| Non-square `112x96` plane | `112x112` or `96x96` | It matches the canonical axial plane, avoids 16 unnecessary columns versus `112x112`, and avoids the field-of-view reduction of `96x96`. Static does not imply square. |
| Orientation-aware letterboxing | Stretch every plane with bilinear interpolation | Letterboxing preserves aspect ratio. Axial is already `112x96`; coronal `96x96` receives 8-pixel top/bottom padding; sagittal `96x112` is transposed. |
| Trilinear images, nearest labels | One interpolation rule | Smooth interpolation is appropriate for intensities; nearest-neighbor preserves discrete segmentation IDs required by Dice and TRE. |
| Mean fusion | Axial only or Gaussian-smoothed fusion | Mean fusion produces the best measured Dice and TRE. Smoothing at sigma 0.75 slightly worsens both and adds work. |
| Segmentation-centroid TRE | No spatial metric or unsupported landmark TRE | private-dataset provides segmentations but no verified manual landmark set. Centroids use available evidence while requiring an explicit limitation statement. |
| Vitis AI 2.5 | An unverified newer toolchain | The repository configuration and generated V4 metadata record `xilinx/vitis-ai-cpu:2.5`, `vitis-ai-pytorch`, `vai_c_xir`, and target `DPUCZDX8G_ISA1_B4096`. |

## 4. Problems Encountered and How They Were Resolved

| Stage | Observed failure | Cause supported by the project record | Resolution and current status |
| --- | --- | --- | --- |
| Early TensorFlow quantization | Calibration flow appeared as zero and FPGA registration failed | The quantization path rebuilt a mismatched architecture and loaded weights by name, losing the trained `flow_scale` behavior | Load the complete saved model and verify the scale layer and calibration range. This is historical; V4 uses the PyTorch path. |
| Early compiler export | Vitis rejected `[1,None,None,7]` inputs | The model had dynamic spatial dimensions | A fixed-shape model was created and output parity was checked. V4 makes the fixed tensor native instead of patching it after training. |
| Early FPGA inference | Flows became excessively large and Dice worsened | A historical inference revision multiplied an already final dequantized flow by 10 | Remove the extra multiplication for the integrated-scale model. `CONTEXT.md` retains both historical notes and identifies the contradiction. |
| Early DPU inputs | Uniform or implausible flow | Quantization scales and tensor ordering were assigned by position | Match tensor semantics by name and respect compiler tensor ordering. |
| Pre-V4 fusion | FPGA fused Dice was below axial Dice | The local and board orientation geometry/fusion contracts were not sufficiently explicit | V4 introduced canonical 3D resampling, fixed orientation transforms, inverse mappings, and matched calibration. V4 DPU mean fusion reaches Dice 0.7647 versus axial 0.7122. |
| FPGA fixed input | Local orientations had different plane sizes | Local PyTorch accepted variable dimensions; the DPU graph did not | Adopt one `112x96x16` external contract with letterboxing/transposition. |
| WSL training smoke run | WSL crashed while loading train and validation data | Too many raw and canonical volumes were held in memory | Use lazy split loading, persistent canonical cache, bounded subject cache, and checkpoint resume. |
| Training duration | An early configuration projected about 1,915 minutes for 120 epochs | Repeated volume preparation and insufficient caching dominated throughput | Increase reusable cached subjects and persist canonical volumes. The completed run reached 1,500 epochs. |
| Board benchmark startup | Every pair was skipped with missing `*_mr.npy` files, followed by a zero-size `np.max` reduction | The board bundle omitted the 18-subject test data, and aggregation assumed at least one successful row | Upload the full test bundle and guard result aggregation against empty inputs. The nine-pair quality run then completed. |
| TRE request | No manual landmarks were available | private-dataset supplied segmentations, not a validated landmark set | Implement segmentation-derived label-centroid TRE and name it precisely. |
| Local Windows sensors | WSL received connection-refused responses from LibreHardwareMonitor | The remote web server was not reachable at WSL localhost | Run LibreHardwareMonitor's web server and discover/use the WSL Windows-host gateway; verify with `--check-only`. |
| Runtime interpretation | 2.5D appeared slow when metric calculation was included | Quality metrics and segmentation warping were inside the old end-to-end window | Add a metric-free inference-to-warp benchmark and report quality evaluation separately. |
| Power comparability | Old board and local energy windows did not match the new inference question | Board values covered a long full-evaluation window; later local values covered inference-to-warp | Implement separate repeated inference-only power windows, common idle calibration, and direct per-inference energy integration. The matched local and board runs are complete. |
| First DPU timing pair | The first metric-free DPU pair took 27.29 s while later pairs were about 13-16 s | The earlier helper warmed one slice rather than the complete pipeline | Warm the complete inference-to-warp path. The final board run confirms stable DPU timing at 13.088 +/- 0.075 s across nine pairs. |

## 5. V4 Training and Calibration

`vxm_2p5d_v4.py` defines the fixed-interface encoder-decoder and shared geometry utilities. The objective combines segmentation Dice loss, masked multimodal MI loss, and masked flow smoothness. Letterbox padding is excluded from relevant loss support.

The completed run used batch size 8, 200 training steps per epoch, 50 fixed validation batches, Adam at `1e-4`, MI weight 0.5, smoothness weight 0.1, caching, and resumable checkpoints. It reached 1,500 epochs; the recorded best validation total loss is `-0.010918`. The negative value is possible because the total includes a negative MI term.

`prepare_calibration_data_v4.py` generated 100 tensors with orientation counts 33 axial, 33 coronal, and 34 sagittal. Each stored tensor is `[16,112,96]`; the compiler-facing batch is `[1,112,96,16]`. Calibration uses the same V4 transforms as training and inference.

`vitis_run_v4.py` uses Vitis AI 2.5 and compiles `vxm_2p5d_pt_v4_canonical` for `DPUCZDX8G_ISA1_B4096`. The canonical export helper is the repository-root `vxm_2p5d_export.py`. A stale duplicate under `Voxelmorph/` should not be treated as canonical.

## 6. Quality Results

### V4 quality across nine pairs

| Configuration | Method | Dice | TRE (mm) | MI | SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| FPGA DPU | Axial | 0.7122 | 2.520 | 0.6043 | 0.7402 |
| FPGA DPU | Mean fused | 0.7647 | 2.025 | 0.5957 | 0.7415 |
| FPGA DPU | Smoothed | 0.7599 | 2.036 | 0.5995 | 0.7398 |
| FPGA ARM CPU | Axial | 0.7160 | 2.500 | 0.6063 | 0.7405 |
| FPGA ARM CPU | Mean fused | 0.7638 | 2.008 | 0.5942 | 0.7416 |
| FPGA ARM CPU | Smoothed | 0.7592 | 2.014 | 0.5988 | 0.7401 |
| Local 3D V2 | Single volume model | 0.7609 | about 2.111 | 0.5772 | 0.7459 |

Mean fusion is the preferred V4 configuration because it gives the highest Dice and lowest TRE. Quantized DPU mean-fused quality remains close to the ARM reference: Dice differs by +0.0009 and TRE by +0.017 mm. The smoothed field offers no measured quality advantage.

The 3D local reference has slightly lower Dice and higher TRE than V4 mean fusion, but slightly higher SSIM. It remains a local comparison only.

## 7. Measurement Methodology

### 7.1 Quality-evaluation scope

The refreshed V4 board/local comparison includes model inference, flow processing, intensity and segmentation warping, Dice, TRE, MI, SSIM, and report work. Its board mean-fused values were:

| Configuration | Model time | Recorded total | Dynamic energy |
| --- | ---: | ---: | ---: |
| FPGA DPU | 1.823 s | 99.198 s | 16.626 J |
| FPGA ARM CPU | 29.472 s | 126.767 s | 27.055 J |

Within this recorded board scope, the DPU model was about 16.2 times faster and dynamic energy was 38.6% lower than ARM CPU. These figures must not be described as inference-only power results.

### 7.2 Local inference-to-warp and power scope

The corrected local benchmark starts when normalized raw in-memory moving/fixed volumes enter model preparation and stops when the warped moving intensity volume is available. Disk I/O, segmentation warping, Dice, TRE, MI, SSIM, plotting, and reports are excluded. Latency is measured without the monitor; power uses a separate repeated execution lasting at least 10 seconds per pair and a shared 20-second idle calibration.

| Configuration | Latency (ms) | Mean monitored power (W) | Absolute energy (J) | Dynamic energy (J) |
| --- | ---: | ---: | ---: | ---: |
| 2.5D CPU | 1021.50 +/- 42.68 | 128.73 +/- 7.83 | 131.44 +/- 9.49 | 63.30 +/- 8.50 |
| 2.5D GPU | 273.31 +/- 17.77 | 122.81 +/- 3.75 | 34.10 +/- 1.68 | 15.57 +/- 1.24 |
| 3D CPU | 656.02 +/- 33.30 | 107.92 +/- 6.68 | 71.15 +/- 6.57 | 27.18 +/- 5.06 |
| 3D GPU | 114.73 +/- 11.42 | 185.87 +/- 9.32 | 21.22 +/- 1.52 | 13.57 +/- 0.91 |

The 3D GPU draws the highest mean monitored power but uses the least absolute energy because it finishes fastest. Energy depends on both power and duration. Desktop totals combine CPU PPT/package and NVIDIA GPU board telemetry; they are not wall-plug totals.

### 7.3 Matched FPGA inference-to-warp measurement

The final board file applies the same boundary as the local benchmark. Latency uses three unmonitored repetitions per pair. Power uses a separate repeated pass of at least 10 seconds, sampled every 0.1 seconds after one shared 20-second idle calibration. PSINTFP represents the ARM/PS rail and INT the DPU/PL rail. Energy is integrated from those samples and normalized per inference; it is not derived from the earlier full-evaluation mean power.

| Configuration | Accelerator/model (ms) | ARM host/non-model (ms) | Total latency (ms) | Mean power (W) | Absolute energy (J) | Dynamic energy (J) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5D DPU | 697.86 | 12390.01 | 13087.88 +/- 74.83 | 1.8475 +/- 0.0109 | 24.243 +/- 0.103 | 2.495 +/- 0.138 |
| 2.5D ARM CPU | 28103.29 | 10155.64 | 38258.93 +/- 213.38 | 2.0941 +/- 0.0094 | 81.136 +/- 0.372 | 16.921 +/- 0.257 |

The corrected full-pipeline warm-up removes the earlier first-pair outlier. DPU inference-to-warp is 2.92 times faster than ARM CPU and uses 70.1% less absolute monitored-rail energy and 85.3% less idle-subtracted dynamic energy. The DPU kernel is only 5.3% of the DPU total, so host preparation, lifting, fusion, resizing, and warping remain the main optimization target.

Desktop and FPGA totals still cover different hardware boundaries. Cross-platform watts and joules are descriptive, not whole-system-equivalent; within-platform comparisons are the strongest claims.
## 8. Findings

1. Fixed `112x96` geometry solved a deployment-contract problem without requiring square images.
2. V4 restored the expected benefit of multi-orientation fusion on the DPU.
3. INT8 DPU quality closely tracks the floating-point ARM reference.
4. Mean fusion is justified; Gaussian smoothing is not.
5. The DPU model stage is fast relative to ARM inference, but the current board pipeline remains host-bound.
6. Local 3D GPU is the fastest measured local inference-to-warp configuration and has the lowest local energy, despite higher mean power.
7. 2.5D remains the FPGA deployment model because its graph and activation demands are far smaller than 3D and it has already been compiled and validated on the board.
8. Cross-platform energy rankings are not defensible without matching sensor coverage. Within-platform comparisons are valid when scope and idle treatment match.

## 9. Limitations

- Nine pairs support an engineering comparison, not broad clinical generalization.
- TRE uses segmentation centroids rather than manually annotated landmarks.
- Canonical resampling may remove fine detail.
- Letterboxing reduces active coronal content.
- DPU/PS rails and desktop CPU/GPU sensors cover different hardware boundaries.
- The current metric-free board timing has a first-pair outlier.
- Direct inference-only FPGA power is pending.
- No 3D FPGA implementation exists.

## 10. Completed Final FPGA Run

The final notebook cell, Inference-only deployment latency, power, and energy, produced inference_pipeline_power_latency_v4.json. The file was validated as schema v2 with nine DPU and nine ARM rows, matching pair indices 0-8 and the same subject identities as the earlier timing run. It is stored at:

    Voxelmorph/artifacts/results/inference_pipeline_breakdown/inference_pipeline_power_latency_v4.json

The energy/power and latency figures now consume this direct file. Future reruns must preserve the same inference-to-warp boundary, warm-up, idle calibration, nine-pair order, and rail definitions so results remain comparable.
## 11. Reproducibility Map

| Purpose | File |
| --- | --- |
| V4 model and geometry | `vxm_2p5d_v4.py` |
| V4 training | `Voxelmorph/2.5D/train_2p5d_pytorchV4.py` and notebook |
| Calibration | `Voxelmorph/2.5D/prepare_calibration_data_v4.py` |
| Vitis AI 2.5 pipeline | `vitis_run_v4.py` |
| Canonical export reference | `vxm_2p5d_export.py` |
| Board quality/inference notebook | `Voxelmorph/fpga_inference_v4.ipynb` |
| Board metric-free timing | `Voxelmorph/fpga_deployment_timing.py` |
| Board matched power measurement | `Voxelmorph/fpga_deployment_power.py` |
| Local matched measurement | `Voxelmorph/measure_local_inference_power_latency.py` |
| Windows sensor runner | `windows_metrics_package/run_windows_benchmark.py` |
| Detailed results | `Voxelmorph/artifacts/results/compare_v4_local_fpga_results/v4_complete_results_analysis.md` |

## Conclusion

The project produced a valid fixed-grid, quantized 2.5D FPGA registration path whose mean-fused quality closely matches its floating-point reference. The main engineering lesson is that geometry, quantization, inference, and measurement definitions form one deployment contract. V4 resolved the earlier fusion inconsistency by making that contract explicit.

The quality conclusion is final for the nine recorded pairs. The local inference-only conclusion is also final for the recorded desktop run. FPGA inference-only power and energy are not final until the latest board notebook produces `inference_pipeline_power_latency_v4.json`.
