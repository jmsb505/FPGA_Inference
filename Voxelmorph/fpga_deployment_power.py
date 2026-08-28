"""Matched inference-only latency, power, and energy on the FPGA board.

This module reuses the validated V4 inference-to-warp pipeline from
``fpga_deployment_timing.py`` and adds separate latency and power passes.
Quality metrics, segmentation warping, disk I/O, plots, and reports are never
inside either measurement window.
"""

from __future__ import annotations

import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fpga_deployment_timing import (
    PAIRS,
    SUBJECTS,
    _load_volume,
    _measure_pair,
)


LATENCY_REPETITIONS = 3
MINIMUM_POWER_WINDOW_S = 10.0
POWER_SAMPLE_INTERVAL_S = 0.10
IDLE_CALIBRATION_S = 20.0

TIMING_FIELDS = (
    "model_inference_ms",
    "accelerator_warp_ms",
    "accelerator_ms",
    "cpu_model_ms",
    "cpu_host_ms",
    "cpu_non_model_ms",
    "total_runtime_ms",
)
POWER_FIELDS = (
    "cpu_power_mean_w",
    "gpu_power_mean_w",
    "power_mean_w",
    "cpu_energy_j_per_inference",
    "gpu_energy_j_per_inference",
    "energy_j_per_inference",
    "cpu_dynamic_energy_j_per_inference",
    "gpu_dynamic_energy_j_per_inference",
    "dynamic_energy_j_per_inference",
    "power_window_s",
    "cpu_power_samples",
    "gpu_power_samples",
)


def _mean_sd(values):
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if array.size > 1 else 0.0,
    )


def _resolve_weights(namespace):
    return namespace["resolve_file"](
        "./2p5d_dense_pt_v4_canonical_best.pth",
        [
            "../trained_weights/2p5d_dense_pt_v4_canonical_best.pth",
            (
                "/home/xilinx/jupyter_notebooks/fpga_run_new/"
                "2p5d_dense_pt_v4_canonical_best.pth"
            ),
        ],
    )


def _validate_warped(warped, moving_raw):
    if warped.shape != moving_raw.shape:
        raise RuntimeError(
            f"Unexpected warped shape {warped.shape}; "
            f"expected {moving_raw.shape}"
        )


def _warm_complete_pipeline(
    namespace,
    moving_raw,
    fixed_raw,
    device,
    weights_path,
):
    _, warped = _measure_pair(
        namespace, moving_raw, fixed_raw, device, weights_path
    )
    _validate_warped(warped, moving_raw)
    del warped
    gc.collect()


def _measure_latency(
    namespace,
    moving_raw,
    fixed_raw,
    device,
    weights_path,
    repetitions,
):
    results = []
    for _ in range(repetitions):
        result, warped = _measure_pair(
            namespace,
            moving_raw,
            fixed_raw,
            device,
            weights_path,
        )
        _validate_warped(warped, moving_raw)
        results.append(result)
        del warped
        gc.collect()

    summary = {
        "model": results[0]["model"],
        "accelerator": results[0]["accelerator"],
        "latency_repetitions": int(repetitions),
    }
    for field in TIMING_FIELDS:
        mean, sd = _mean_sd([result[field] for result in results])
        summary[field] = mean
        summary[f"{field}_sd_within_pair"] = sd
    return summary


def _calibrate_idle(namespace, duration_s, sample_interval_s):
    monitor_type = namespace.get("PowerMonitor")
    if monitor_type is None:
        raise RuntimeError(
            "PowerMonitor is unavailable. Run the notebook utility cell "
            "before this benchmark cell."
        )

    print(
        f"Calibrating common board idle power for {duration_s:.1f} s..."
    )
    with monitor_type(
        "fpga",
        sample_interval_s=sample_interval_s,
        idle_dpu_w=0.0,
        idle_cpu_w=0.0,
    ) as monitor:
        time.sleep(duration_s)
    result = monitor.result()
    cpu_samples = np.asarray(
        getattr(monitor, "cpu_samples_w", []), dtype=np.float64
    )
    dpu_samples = np.asarray(
        getattr(monitor, "gpu_samples_w", []), dtype=np.float64
    )
    if cpu_samples.size < 2 or dpu_samples.size < 2:
        raise RuntimeError(
            "Idle calibration did not capture both PSINTFP and INT rails. "
            "Confirm pynq.get_rails() exposes both sensors."
        )

    cpu_mean, cpu_sd = _mean_sd(cpu_samples)
    dpu_mean, dpu_sd = _mean_sd(dpu_samples)
    calibration = {
        "duration_s": float(result["power_wall_time_s"]),
        "sample_interval_s": float(sample_interval_s),
        "arm_cpu": {
            "sensor": "PSINTFP monitored ARM/PS rail",
            "mean_w": cpu_mean,
            "sd_w": cpu_sd,
            "peak_w": float(cpu_samples.max()),
            "samples": int(cpu_samples.size),
        },
        "dpu": {
            "sensor": "INT monitored DPU/PL rail",
            "mean_w": dpu_mean,
            "sd_w": dpu_sd,
            "peak_w": float(dpu_samples.max()),
            "samples": int(dpu_samples.size),
        },
    }
    print(
        "  Idle rails: "
        f"ARM/PS={cpu_mean:.4f} W, DPU/PL={dpu_mean:.4f} W"
    )
    return calibration


def _sum_known(parts, key):
    values = [
        float(part[key]) for part in parts if part.get(key) is not None
    ]
    return float(sum(values)) if values else None


def _per_inference(value, repetitions):
    return None if value is None else float(value / repetitions)


def _measure_power(
    namespace,
    moving_raw,
    fixed_raw,
    device,
    weights_path,
    idle_cpu_w,
    idle_dpu_w,
    minimum_window_s,
    sample_interval_s,
):
    monitor_type = namespace["PowerMonitor"]
    parts = []
    cpu_samples = []
    dpu_samples = []
    repetitions = 0
    started = time.perf_counter()

    while repetitions == 0 or time.perf_counter() - started < minimum_window_s:
        with monitor_type(
            device,
            sample_interval_s=sample_interval_s,
            idle_dpu_w=idle_dpu_w,
            idle_cpu_w=idle_cpu_w,
        ) as monitor:
            _, warped = _measure_pair(
                namespace,
                moving_raw,
                fixed_raw,
                device,
                weights_path,
            )
        _validate_warped(warped, moving_raw)
        parts.append(monitor.result())
        cpu_samples.extend(getattr(monitor, "cpu_samples_w", []))
        dpu_samples.extend(getattr(monitor, "gpu_samples_w", []))
        repetitions += 1
        del warped
        gc.collect()

    wall_time_s = float(
        sum(float(part.get("power_wall_time_s") or 0.0) for part in parts)
    )
    cpu_energy = _sum_known(parts, "cpu_energy_j")
    dpu_energy = _sum_known(parts, "gpu_energy_j")
    total_energy = _sum_known(parts, "energy_j")
    cpu_dynamic = _sum_known(parts, "cpu_dynamic_energy_j")
    dpu_dynamic = _sum_known(parts, "gpu_dynamic_energy_j")
    total_dynamic = _sum_known(parts, "dynamic_energy_j")
    cpu_array = np.asarray(cpu_samples, dtype=np.float64)
    dpu_array = np.asarray(dpu_samples, dtype=np.float64)
    if cpu_array.size < 2 or dpu_array.size < 2:
        raise RuntimeError(
            "The power window did not capture both PSINTFP and INT rails."
        )

    cpu_mean, cpu_sd = _mean_sd(cpu_array)
    dpu_mean, dpu_sd = _mean_sd(dpu_array)
    total_mean = (
        None
        if total_energy is None or wall_time_s <= 0.0
        else float(total_energy / wall_time_s)
    )
    result = {
        "power_repetitions": int(repetitions),
        "power_window_s": wall_time_s,
        "cpu_power_mean_w": cpu_mean,
        "cpu_power_sd_w": cpu_sd,
        "cpu_power_peak_w": float(cpu_array.max()),
        "cpu_power_samples": int(cpu_array.size),
        "gpu_power_mean_w": dpu_mean,
        "gpu_power_sd_w": dpu_sd,
        "gpu_power_peak_w": float(dpu_array.max()),
        "gpu_power_samples": int(dpu_array.size),
        "power_mean_w": total_mean,
        "cpu_energy_j_per_inference": _per_inference(
            cpu_energy, repetitions
        ),
        "gpu_energy_j_per_inference": _per_inference(
            dpu_energy, repetitions
        ),
        "energy_j_per_inference": _per_inference(
            total_energy, repetitions
        ),
        "cpu_dynamic_energy_j_per_inference": _per_inference(
            cpu_dynamic, repetitions
        ),
        "gpu_dynamic_energy_j_per_inference": _per_inference(
            dpu_dynamic, repetitions
        ),
        "dynamic_energy_j_per_inference": _per_inference(
            total_dynamic, repetitions
        ),
        "idle_cpu_power_w": float(idle_cpu_w),
        "idle_gpu_power_w": float(idle_dpu_w),
    }

    # Explicit board-resource aliases avoid treating the monitored INT rail as
    # a GPU while retaining the generic keys used by earlier result readers.
    result.update(
        arm_cpu_power_mean_w=result["cpu_power_mean_w"],
        arm_cpu_power_sd_w=result["cpu_power_sd_w"],
        arm_cpu_power_peak_w=result["cpu_power_peak_w"],
        arm_cpu_power_samples=result["cpu_power_samples"],
        dpu_power_mean_w=result["gpu_power_mean_w"],
        dpu_power_sd_w=result["gpu_power_sd_w"],
        dpu_power_peak_w=result["gpu_power_peak_w"],
        dpu_power_samples=result["gpu_power_samples"],
        arm_cpu_energy_j_per_inference=result[
            "cpu_energy_j_per_inference"
        ],
        dpu_energy_j_per_inference=result[
            "gpu_energy_j_per_inference"
        ],
        arm_cpu_dynamic_energy_j_per_inference=result[
            "cpu_dynamic_energy_j_per_inference"
        ],
        dpu_dynamic_energy_j_per_inference=result[
            "gpu_dynamic_energy_j_per_inference"
        ],
        idle_arm_cpu_power_w=result["idle_cpu_power_w"],
        idle_dpu_power_w=result["idle_gpu_power_w"],
    )
    return result


def _aggregate(rows, model):
    selected = [row for row in rows if row["model"] == model]
    result = {}
    for field in TIMING_FIELDS + POWER_FIELDS:
        mean, sd = _mean_sd([row[field] for row in selected])
        result[f"{field}_mean"] = mean
        result[f"{field}_sd"] = sd
    result.update(
        pair_count=len(selected),
        idle_cpu_power_w=float(selected[0]["idle_cpu_power_w"]),
        idle_gpu_power_w=float(selected[0]["idle_gpu_power_w"]),
        idle_arm_cpu_power_w=float(
            selected[0]["idle_arm_cpu_power_w"]
        ),
        idle_dpu_power_w=float(selected[0]["idle_dpu_power_w"]),
    )
    return result


def run_benchmark(
    namespace,
    output_path="inference_pipeline_power_latency_v4.json",
    latency_repetitions=LATENCY_REPETITIONS,
    minimum_power_window_s=MINIMUM_POWER_WINDOW_S,
    power_sample_interval_s=POWER_SAMPLE_INTERVAL_S,
    idle_calibration_s=IDLE_CALIBRATION_S,
):
    if latency_repetitions < 1:
        raise ValueError("latency_repetitions must be at least one")
    if minimum_power_window_s <= 0.0:
        raise ValueError("minimum_power_window_s must be positive")
    if power_sample_interval_s <= 0.0:
        raise ValueError("power_sample_interval_s must be positive")
    if idle_calibration_s <= 0.0:
        raise ValueError("idle_calibration_s must be positive")

    weights_path = _resolve_weights(namespace)
    warm_moving = _load_volume(namespace, SUBJECTS[PAIRS[0][0]], "mr")
    warm_fixed = _load_volume(namespace, SUBJECTS[PAIRS[0][1]], "ct")
    for device in ("fpga", "cpu"):
        print(f"Warming the complete {device.upper()} inference path...")
        _warm_complete_pipeline(
            namespace,
            warm_moving,
            warm_fixed,
            device,
            weights_path,
        )

    idle = _calibrate_idle(
        namespace,
        duration_s=idle_calibration_s,
        sample_interval_s=power_sample_interval_s,
    )
    idle_cpu_w = float(idle["arm_cpu"]["mean_w"])
    idle_dpu_w = float(idle["dpu"]["mean_w"])

    rows = []
    for device in ("fpga", "cpu"):
        print(f"\nInference-only benchmark: {device.upper()}")
        for pair_index, (moving_idx, fixed_idx) in enumerate(PAIRS):
            moving_id = SUBJECTS[moving_idx]
            fixed_id = SUBJECTS[fixed_idx]
            moving_raw = _load_volume(namespace, moving_id, "mr")
            fixed_raw = _load_volume(namespace, fixed_id, "ct")

            result = _measure_latency(
                namespace,
                moving_raw,
                fixed_raw,
                device,
                weights_path,
                repetitions=latency_repetitions,
            )
            result.update(
                _measure_power(
                    namespace,
                    moving_raw,
                    fixed_raw,
                    device,
                    weights_path,
                    idle_cpu_w=idle_cpu_w,
                    idle_dpu_w=idle_dpu_w,
                    minimum_window_s=minimum_power_window_s,
                    sample_interval_s=power_sample_interval_s,
                )
            )
            result.update(
                pair_index=pair_index,
                moving_idx=moving_idx,
                fixed_idx=fixed_idx,
                moving_id=moving_id,
                fixed_id=fixed_id,
            )
            rows.append(result)
            print(
                f"  Pair {pair_index + 1}/{len(PAIRS)}: "
                f"accelerator={result['accelerator_ms']:.1f} ms, "
                f"ARM CPU={result['cpu_host_ms']:.1f} ms, "
                f"total={result['total_runtime_ms']:.1f} ms, "
                f"power={result['power_mean_w']:.3f} W, "
                f"energy={result['energy_j_per_inference']:.3f} J"
            )
            gc.collect()

    model_names = ("2.5d_fused_dpu", "2.5d_fused_arm_cpu")
    models = {model: _aggregate(rows, model) for model in model_names}
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "platform": platform.platform(),
        "pair_count": len(PAIRS),
        "settings": {
            "latency_repetitions": int(latency_repetitions),
            "minimum_power_window_s": float(minimum_power_window_s),
            "power_sample_interval_s": float(power_sample_interval_s),
            "idle_calibration_s": float(idle_calibration_s),
        },
        "scope": {
            "start": (
                "normalized raw in-memory moving/fixed volumes enter model "
                "preparation"
            ),
            "end": (
                "warped moving intensity volume is available in host memory"
            ),
            "included": (
                "model-input preparation, transfers/quantization, model "
                "execution, all three orientations, lifting/fusion, flow "
                "resizing, and intensity-volume warping"
            ),
            "excluded": (
                "disk I/O, segmentation warping, Dice, TRE, MI, SSIM, "
                "plotting, and report generation"
            ),
            "latency_measurement": (
                "repeated without power sampling; each pair stores its mean "
                "and within-pair SD"
            ),
            "power_measurement": (
                "a separate execution of the identical inference-to-warp "
                "pipeline, repeated until the minimum sampling window"
            ),
            "cpu_definition": (
                "DPU: total wall minus DPU runner time; ARM CPU: total wall"
            ),
            "accelerator_definition": (
                "DPU execute_async/wait only; ARM CPU has no accelerator "
                "component"
            ),
        },
        "sensor_coverage": {
            "arm_cpu": "PSINTFP monitored ARM/PS rail",
            "dpu": "INT monitored DPU/PL rail",
            "total": (
                "sum of PSINTFP and INT; this is not whole-board wall power"
            ),
            "dynamic_energy": (
                "each rail is idle-subtracted independently using one shared "
                "20-second idle calibration and clamped at zero"
            ),
        },
        "idle_calibration": idle,
        "models": models,
        "pairs": rows,
    }

    destination = Path(output_path)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nNine-pair averages:")
    print(json.dumps(models, indent=2))
    print(f"Wrote {destination.resolve()}")
    return payload
