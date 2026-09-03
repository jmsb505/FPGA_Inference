"""Measure 3D GPU runtime, CPU/GPU energy, and power on the nine test pairs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
VOX_DIR = REPO_ROOT / "Voxelmorph"
SOURCE_NOTEBOOK = VOX_DIR / "compare_2p5d_v4_fusion_3d_v2_test.ipynb"
OUTPUT_DIR = (
    VOX_DIR
    / "artifacts"
    / "results"
    / "compare_2p5d_gpu_dpu_3d_gpu_resources"
)
ROWS_PATH = OUTPUT_DIR / "3d_gpu_power_rows.json"
SUMMARY_PATH = OUTPUT_DIR / "3d_gpu_power_summary.json"


def load_measurement_namespace() -> dict:
    """Load the established data, metric, TRE, and local-power definitions."""
    os.environ.setdefault("V4_MEASURE_LOCAL_POWER", "1")
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict = {}
    original_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        for cell_index in (1, 2, 3, 4):
            exec("".join(notebook["cells"][cell_index]["source"]), namespace)
    finally:
        os.chdir(original_cwd)
    return namespace


def query_gpu_state() -> dict:
    command = [
        "/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    process = subprocess.run(
        command, capture_output=True, text=True, timeout=5.0, check=False
    )
    if process.returncode != 0 or not process.stdout.strip():
        return {}
    values = [part.strip() for part in process.stdout.splitlines()[0].split(",")]
    if len(values) != 4:
        return {}
    return {
        "power_w": float(values[0]),
        "utilization_percent": float(values[1]),
        "memory_used_mb": float(values[2]),
        "temperature_c": float(values[3]),
    }


def preflight(
    namespace: dict,
    max_gpu_utilization: float,
    max_gpu_memory_mb: float,
    max_cpu_power_w: float,
    allow_busy: bool,
) -> dict:
    gpu_samples = []
    cpu_samples = []
    for _ in range(8):
        state = query_gpu_state()
        if state:
            gpu_samples.append(state)
        cpu_power = namespace["query_local_cpu_power_w"]()
        if cpu_power is not None:
            cpu_samples.append(float(cpu_power))
        time.sleep(0.25)

    if not gpu_samples:
        raise RuntimeError("Could not read NVIDIA GPU state through nvidia-smi")
    if not cpu_samples:
        raise RuntimeError(
            "CPU package power is unavailable. Start LibreHardwareMonitor's "
            "remote web server before benchmarking."
        )

    result = {
        "gpu_utilization_percent_mean": float(
            np.mean([sample["utilization_percent"] for sample in gpu_samples])
        ),
        "gpu_memory_used_mb_mean": float(
            np.mean([sample["memory_used_mb"] for sample in gpu_samples])
        ),
        "gpu_power_w_mean": float(
            np.mean([sample["power_w"] for sample in gpu_samples])
        ),
        "cpu_power_w_mean": float(np.mean(cpu_samples)),
        "sample_count": len(gpu_samples),
    }
    busy_reasons = []
    if result["gpu_utilization_percent_mean"] > max_gpu_utilization:
        busy_reasons.append(
            f"GPU utilization {result['gpu_utilization_percent_mean']:.1f}% "
            f"> {max_gpu_utilization:.1f}%"
        )
    if result["gpu_memory_used_mb_mean"] > max_gpu_memory_mb:
        busy_reasons.append(
            f"GPU memory {result['gpu_memory_used_mb_mean']:.0f} MB "
            f"> {max_gpu_memory_mb:.0f} MB"
        )
    if result["cpu_power_w_mean"] > max_cpu_power_w:
        busy_reasons.append(
            f"CPU package power {result['cpu_power_w_mean']:.1f} W "
            f"> {max_cpu_power_w:.1f} W"
        )
    result["busy_reasons"] = busy_reasons
    if busy_reasons and not allow_busy:
        raise RuntimeError(
            "Preflight rejected a contaminated idle baseline: "
            + "; ".join(busy_reasons)
            + ". Stop other workloads and rerun, or use --allow-busy only "
            "if you intentionally accept invalid comparative telemetry."
        )
    return result


@torch.no_grad()
def measure_pair(
    namespace: dict,
    model,
    raw: tuple,
    runtime_device,
    idle_power: dict,
    inference_repetitions: int,
    postprocess_repetitions: int,
) -> dict:
    moving, fixed, moving_seg, fixed_seg = raw
    moving_tensor = namespace["preprocess_volume_3d"](
        moving, is_seg=False, device_override=runtime_device
    )
    fixed_tensor = namespace["preprocess_volume_3d"](
        fixed, is_seg=False, device_override=runtime_device
    )

    def inference_stage():
        torch.cuda.synchronize(runtime_device)
        start = time.perf_counter()
        model_output = model(moving_tensor, fixed_tensor)
        flow_downsampled = (
            model_output[-1]
            if isinstance(model_output, (tuple, list))
            else model_output
        )
        torch.cuda.synchronize(runtime_device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return flow_downsampled, elapsed_ms

    inference_result, inference_power, inference_ms = namespace[
        "measure_local_stage"
    ](
        inference_stage,
        runtime_device,
        idle_power,
        repetitions=inference_repetitions,
        duration_getter=lambda result: result[1],
    )
    flow_downsampled = inference_result[0]

    def postprocess_stage():
        start = time.perf_counter()
        flow_raw = namespace["upsample_3d_flow_to_raw"](
            flow_downsampled[0].detach().cpu().numpy(), moving.shape
        )
        moved_raw = namespace["warp_volume_3d"](
            moving, flow_raw, mode="bilinear", runtime_device=runtime_device
        )
        warped_segmentation = namespace["warp_volume_3d"](
            moving_seg.astype(np.float32),
            flow_raw,
            mode="nearest",
            runtime_device=runtime_device,
        ).astype(np.int16)
        metrics = namespace["summarize_registration"](
            moving,
            fixed,
            moving_seg,
            fixed_seg,
            flow_raw,
            moved_raw,
            warped_segmentation,
        )
        metrics.update(
            namespace["label_centroid_tre_metrics"](
                moving_seg, warped_segmentation, fixed_seg
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return metrics, elapsed_ms

    postprocess_result, postprocess_power, postprocess_ms = namespace[
        "measure_local_stage"
    ](
        postprocess_stage,
        runtime_device,
        idle_power,
        repetitions=postprocess_repetitions,
        duration_getter=lambda result: result[1],
    )
    metrics = dict(postprocess_result[0])
    metrics.update(
        model_inference_ms=float(inference_ms),
        postprocess_ms=float(postprocess_ms),
        total_runtime_ms=float(inference_ms + postprocess_ms),
        inference_power_repetitions=int(inference_repetitions),
        postprocess_power_repetitions=int(postprocess_repetitions),
        measurement_boundary=(
            "GPU model forward; then flow lifting, image/segmentation warping, "
            "Dice/MI/SSIM, and segmentation-derived TRE"
        ),
    )
    metrics.update(
        namespace["combine_local_power_measurements"](
            [inference_power, postprocess_power]
        )
    )
    return metrics


def aggregate_rows(rows: list[dict]) -> dict:
    keys = (
        "model_inference_ms",
        "postprocess_ms",
        "total_runtime_ms",
        "power_wall_time_s",
        "cpu_energy_j",
        "gpu_energy_j",
        "energy_j",
        "cpu_dynamic_energy_j",
        "gpu_dynamic_energy_j",
        "dynamic_energy_j",
        "cpu_power_mean_w",
        "gpu_power_mean_w",
        "power_mean_w",
        "cpu_power_peak_w",
        "gpu_power_peak_w",
        "process_rss_peak_mb",
        "process_rss_delta_mb",
        "dice_after",
        "tre_mm",
        "mi_after",
        "ssim_deformed_fixed",
    )
    result = {"pair_count": len(rows), "metrics": {}}
    for key in keys:
        values = np.asarray(
            [row[key] for row in rows if row.get(key) is not None],
            dtype=np.float64,
        )
        result["metrics"][key] = {
            "mean": None if values.size == 0 else float(values.mean()),
            "sample_sd": (
                None
                if values.size == 0
                else float(values.std(ddof=1))
                if values.size > 1
                else 0.0
            ),
            "count": int(values.size),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-limit", type=int, default=None)
    parser.add_argument("--inference-repetitions", type=int, default=25)
    parser.add_argument("--postprocess-repetitions", type=int, default=1)
    parser.add_argument("--max-gpu-utilization", type=float, default=10.0)
    parser.add_argument("--max-gpu-memory-mb", type=float, default=3000.0)
    parser.add_argument("--max-cpu-power-w", type=float, default=80.0)
    parser.add_argument("--allow-busy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    namespace = load_measurement_namespace()
    preflight_result = preflight(
        namespace,
        max_gpu_utilization=args.max_gpu_utilization,
        max_gpu_memory_mb=args.max_gpu_memory_mb,
        max_cpu_power_w=args.max_cpu_power_w,
        allow_busy=args.allow_busy,
    )
    print("Preflight:", preflight_result)

    runtime_device = torch.device("cuda")
    test_dataset = namespace["load_split"]("test", None)
    test_pairs = namespace["build_eval_pairs"](
        len(test_dataset),
        max_pairs=args.pair_limit,
        seed=namespace["PAIR_SEED"],
    )
    model = namespace["VxmDense3DV2"]().to(runtime_device).eval()
    state = torch.load(
        namespace["V3D_WEIGHTS"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(state.get("state_dict", state))

    first_values = namespace["canonical_pair"](test_dataset, *test_pairs[0])
    first_raw = first_values[:4]
    namespace["run_3d_pair"](model, *first_raw, runtime_device)
    torch.cuda.synchronize(runtime_device)

    print(
        f"Calibrating idle power for "
        f"{namespace['LOCAL_POWER_IDLE_SECONDS']:.1f}s..."
    )
    idle_power = namespace["calibrate_local_idle_power"](runtime_device)
    if idle_power.get("idle_cpu_w") is None or idle_power.get("idle_gpu_w") is None:
        raise RuntimeError(f"Incomplete idle calibration: {idle_power}")
    print("Idle calibration:", idle_power)

    rows = []
    for pair_index, (moving_index, fixed_index) in enumerate(test_pairs):
        values = namespace["canonical_pair"](
            test_dataset, moving_index, fixed_index
        )
        metrics = measure_pair(
            namespace,
            model,
            values[:4],
            runtime_device,
            idle_power,
            inference_repetitions=args.inference_repetitions,
            postprocess_repetitions=args.postprocess_repetitions,
        )
        metrics.update(
            device="cuda",
            method="3d_gpu_measured",
            pair_index=int(pair_index),
            moving_idx=int(moving_index),
            fixed_idx=int(fixed_index),
        )
        rows.append(metrics)
        print(
            f"Pair {pair_index + 1}/{len(test_pairs)}: "
            f"runtime={metrics['total_runtime_ms'] / 1000.0:.3f}s, "
            f"energy={metrics.get('energy_j'):.3f}J, "
            f"dynamic={metrics.get('dynamic_energy_j'):.3f}J"
        )

    payload = {
        "measurement": "3D GPU matched-pipeline power benchmark",
        "pair_count": len(rows),
        "preflight": preflight_result,
        "idle_calibration": idle_power,
        "inference_repetitions": args.inference_repetitions,
        "postprocess_repetitions": args.postprocess_repetitions,
        "rows": rows,
    }
    aggregate = aggregate_rows(rows)
    aggregate.update(
        preflight=preflight_result,
        idle_calibration=idle_power,
        measurement_boundary=rows[0]["measurement_boundary"],
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROWS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"Wrote {ROWS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
