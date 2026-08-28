# Project Documentation Index

**Updated:** 2026-08-28

## Current documents

| Document | Purpose |
| --- | --- |
| `README.md` | Public entry point, active commands, and asset policy |
| `PROJECT_EXPERIMENT_REPORT.md` | Project narrative, experimental setup, results, limitations, and conclusions |
| `CONTEXT.md` | V4 runbook plus preserved debugging and design history |
| `AGENTS.md` | Contributor workflow, measurement rules, and review expectations |
| `Voxelmorph/artifacts/results/compare_v4_local_fpga_results/v4_complete_results_analysis.md` | Detailed quality, latency, power, and energy interpretation |
| `Voxelmorph/artifacts/results/compare_v4_local_fpga_results/v4_local_fpga_comparison_summary.md` | Concise V4 platform comparison |
| `Voxelmorph/artifacts/results/compare_2p5d_v4_fusion_3d_v2_test/benchmark_summary_v4_local.md` | Local 2.5D-versus-3D quality summary |
| `windows_metrics_package/README_WINDOWS_METRICS.md` | Windows/WSL sensor setup and benchmark instructions |

Superseded V2 details are summarized in `CONTEXT.md`; large historical outputs and transfer archives are intentionally excluded from Git.

## Canonical result sources

- `Voxelmorph/benchmark_results_comparison_v4.json`: FPGA ARM CPU and DPU quality.
- `Voxelmorph/artifacts/results/compare_2p5d_v4_fusion_3d_v2_test/benchmark_results_comparison_v4_local.json`: local V4 quality.
- `Voxelmorph/artifacts/results/compare_2p5d_v4_fusion_3d_v2_test/v4_vs_3d_rows.json`: aligned V4-versus-3D rows.
- `Voxelmorph/artifacts/results/inference_only_local_power_latency/local_2p5d_3d_cpu_gpu_9_pairs.json`: local latency, power, and energy.
- `Voxelmorph/artifacts/results/inference_pipeline_breakdown/inference_pipeline_power_latency_v4.json`: final FPGA inference-only measurements.
- `Voxelmorph/artifacts/results/presentation_inputs/presentation_resource_metrics_v4.json`: compact parameter-count and CUDA-memory inputs.
- `Voxelmorph/artifacts/results/compare_v4_local_fpga_results/v4_platform_method_summary.csv`: presentation-level platform summary.

## Truth and measurement rules

- Every numeric claim must name its source JSON or table.
- Quality, inference-to-warp latency, and inference-only power are separate scopes.
- Local inference-only results are directly measured for 2.5D/3D CPU/GPU.
- Direct FPGA inference-only latency, power, and energy are measured for DPU and ARM CPU.
- Desktop sensor totals and FPGA rail totals are not whole-system-equivalent.
- TRE is segmentation-derived anatomical-label-centroid TRE.

The final board file has schema v2, nine DPU rows, nine ARM CPU rows, matching pair identities, and finite latency, power, energy, and idle-baseline aggregates. Future reruns must retain the same inference-to-warp boundary, full-pipeline warm-up, 20-second idle calibration, 0.1-second sampling, monitored rails, and nine-pair order.
