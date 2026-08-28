"""Measure metric-free local latency, power, and energy on CPU and GPU."""

from __future__ import annotations

import gc
import json
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
import torch

from measure_3d_gpu_power import load_measurement_namespace
from measure_inference_pipeline_breakdown import load_raw_pair


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = (
    REPO_ROOT
    / "Voxelmorph"
    / "artifacts"
    / "results"
    / "inference_only_local_power_latency"
    / "local_2p5d_3d_cpu_gpu_9_pairs.json"
)
POWER_WINDOW_SECONDS = float(
    os.environ.get("INFERENCE_POWER_WINDOW_SECONDS", "10.0")
)
POWER_SAMPLE_INTERVAL_SECONDS = float(
    os.environ.get("INFERENCE_POWER_SAMPLE_INTERVAL_SECONDS", "0.20")
)
IDLE_CALIBRATION_SECONDS = float(
    os.environ.get("INFERENCE_IDLE_CALIBRATION_SECONDS", "8.0")
)
LATENCY_REPETITIONS = int(
    os.environ.get("INFERENCE_LATENCY_REPETITIONS", "3")
)
PAIR_LIMIT = int(os.environ.get("INFERENCE_PAIR_LIMIT", "0"))


def _load_power_queries():
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from windows_metrics_package.run_windows_benchmark import (
        query_cpu_power_w,
        query_gpu_power_w,
    )

    return query_cpu_power_w, query_gpu_power_w


QUERY_CPU_POWER_W, QUERY_GPU_POWER_W = _load_power_queries()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class SampledPowerMonitor:
    """Sample CPU package and GPU board power over one repeated workload."""

    def __init__(self, sample_interval_s: float) -> None:
        self.sample_interval_s = float(sample_interval_s)
        self.cpu_samples: list[tuple[float, float]] = []
        self.gpu_samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None

    def _sample_once(self) -> None:
        timestamp = time.perf_counter()
        cpu = QUERY_CPU_POWER_W()
        gpu = QUERY_GPU_POWER_W()
        if cpu is not None:
            self.cpu_samples.append((timestamp, float(cpu)))
        if gpu is not None:
            self.gpu_samples.append((timestamp, float(gpu)))

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            self._sample_once()

    def __enter__(self):
        self._sample_once()
        self.start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end_time = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._sample_once()
        return False

    @staticmethod
    def _summarize(samples: list[tuple[float, float]]) -> dict:
        values = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        if not values.size:
            return {
                "mean_w": None,
                "sd_w": None,
                "peak_w": None,
                "samples": 0,
            }
        return {
            "mean_w": float(values.mean()),
            "sd_w": (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            ),
            "peak_w": float(values.max()),
            "samples": int(values.size),
        }

    def result(self) -> dict:
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("Power monitor did not complete")
        return {
            "wall_time_s": float(self.end_time - self.start_time),
            "cpu": self._summarize(self.cpu_samples),
            "gpu": self._summarize(self.gpu_samples),
        }


def calibrate_idle() -> dict:
    with SampledPowerMonitor(POWER_SAMPLE_INTERVAL_SECONDS) as monitor:
        time.sleep(IDLE_CALIBRATION_SECONDS)
    result = monitor.result()
    if result["cpu"]["mean_w"] is None:
        raise RuntimeError("LibreHardwareMonitor CPU power is unavailable")
    if result["gpu"]["mean_w"] is None:
        raise RuntimeError("NVIDIA GPU power is unavailable")
    return result


@torch.no_grad()
def run_v4_pipeline(
    namespace: dict,
    model,
    moving_raw: np.ndarray,
    fixed_raw: np.ndarray,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    synchronize(device)
    total_started = time.perf_counter()
    moving_c = namespace["normalize_volume_contract"](
        namespace["resample_volume"](moving_raw)
    )
    fixed_c = namespace["normalize_volume_contract"](
        namespace["resample_volume"](fixed_raw)
    )

    flows = {}
    model_ms = 0.0
    for orientation in ("axial", "coronal", "sagittal"):
        flow, elapsed_ms = namespace["infer_v4_orientation"](
            model, moving_c, fixed_c, orientation, device
        )
        flows[orientation] = flow
        model_ms += float(elapsed_ms)

    axial = namespace["lift_axial"](flows["axial"], moving_c.shape)
    coronal = namespace["lift_coronal"](flows["coronal"], moving_c.shape)
    sagittal = namespace["lift_sagittal"](flows["sagittal"], moving_c.shape)
    fused = namespace["fuse_fields"](axial, coronal, sagittal)
    raw_flow = namespace["canonical_field_to_raw"](fused, moving_raw.shape)

    synchronize(device)
    warp_started = time.perf_counter()
    warped = namespace["warp_volume_3d"](
        moving_raw,
        raw_flow,
        mode="bilinear",
        runtime_device=device,
    )
    synchronize(device)
    warp_ms = (time.perf_counter() - warp_started) * 1000.0
    total_ms = (time.perf_counter() - total_started) * 1000.0
    return (
        {
            "model_inference_ms": model_ms,
            "warp_ms": warp_ms,
            "total_runtime_ms": total_ms,
        },
        warped,
    )


@torch.no_grad()
def run_3d_pipeline(
    namespace: dict,
    model,
    moving_raw: np.ndarray,
    fixed_raw: np.ndarray,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    synchronize(device)
    total_started = time.perf_counter()
    moving_tensor = namespace["preprocess_volume_3d"](
        moving_raw, is_seg=False, device_override=device
    )
    fixed_tensor = namespace["preprocess_volume_3d"](
        fixed_raw, is_seg=False, device_override=device
    )

    synchronize(device)
    model_started = time.perf_counter()
    model_output = model(moving_tensor, fixed_tensor)
    flow_downsampled = (
        model_output[-1]
        if isinstance(model_output, (tuple, list))
        else model_output
    )
    synchronize(device)
    model_ms = (time.perf_counter() - model_started) * 1000.0

    raw_flow = namespace["upsample_3d_flow_to_raw"](
        flow_downsampled[0].detach().cpu().numpy(), moving_raw.shape
    )
    synchronize(device)
    warp_started = time.perf_counter()
    warped = namespace["warp_volume_3d"](
        moving_raw,
        raw_flow,
        mode="bilinear",
        runtime_device=device,
    )
    synchronize(device)
    warp_ms = (time.perf_counter() - warp_started) * 1000.0
    total_ms = (time.perf_counter() - total_started) * 1000.0
    return (
        {
            "model_inference_ms": model_ms,
            "warp_ms": warp_ms,
            "total_runtime_ms": total_ms,
        },
        warped,
    )


def _load_v4_model(namespace: dict, device: torch.device):
    return (
        namespace["load_board_cpu_reference"](namespace["V4_WEIGHTS"])
        .to(device)
        .eval()
    )


def _load_3d_model(namespace: dict, device: torch.device):
    model = namespace["VxmDense3DV2"]().to(device).eval()
    state = torch.load(
        namespace["V3D_WEIGHTS"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(state.get("state_dict", state))
    return model


def _mean_sd(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if array.size > 1 else 0.0,
    )


def _power_fields(
    power: dict,
    idle: dict,
    repetitions: int,
) -> dict:
    wall = float(power["wall_time_s"])
    cpu_mean = float(power["cpu"]["mean_w"])
    gpu_mean = float(power["gpu"]["mean_w"])
    idle_cpu = float(idle["cpu"]["mean_w"])
    idle_gpu = float(idle["gpu"]["mean_w"])
    cpu_energy = cpu_mean * wall
    gpu_energy = gpu_mean * wall
    cpu_dynamic = max(cpu_mean - idle_cpu, 0.0) * wall
    gpu_dynamic = max(gpu_mean - idle_gpu, 0.0) * wall
    return {
        "power_window_s": wall,
        "power_repetitions": int(repetitions),
        "cpu_power_mean_w": cpu_mean,
        "cpu_power_sd_w": power["cpu"]["sd_w"],
        "cpu_power_peak_w": power["cpu"]["peak_w"],
        "cpu_power_samples": power["cpu"]["samples"],
        "gpu_power_mean_w": gpu_mean,
        "gpu_power_sd_w": power["gpu"]["sd_w"],
        "gpu_power_peak_w": power["gpu"]["peak_w"],
        "gpu_power_samples": power["gpu"]["samples"],
        "power_mean_w": cpu_mean + gpu_mean,
        "cpu_energy_j_per_inference": cpu_energy / repetitions,
        "gpu_energy_j_per_inference": gpu_energy / repetitions,
        "energy_j_per_inference": (
            cpu_energy + gpu_energy
        ) / repetitions,
        "cpu_dynamic_energy_j_per_inference": cpu_dynamic / repetitions,
        "gpu_dynamic_energy_j_per_inference": gpu_dynamic / repetitions,
        "dynamic_energy_j_per_inference": (
            cpu_dynamic + gpu_dynamic
        ) / repetitions,
    }


def benchmark_configuration(
    namespace: dict,
    dataset,
    pairs: list[tuple[int, int]],
    config_name: str,
    pipeline,
    model,
    device: torch.device,
) -> tuple[list[dict], dict]:
    warm_moving, warm_fixed = load_raw_pair(namespace, dataset, *pairs[0])
    print(f"\nFull-pipeline warm-up: {config_name}", flush=True)
    pipeline(
        namespace, model, warm_moving, warm_fixed, device
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    time.sleep(2.0)
    print(
        f"Idle calibration: {config_name} "
        f"({IDLE_CALIBRATION_SECONDS:.1f}s)",
        flush=True,
    )
    idle = calibrate_idle()
    print(
        "  Idle:",
        {
            "cpu_w": round(idle["cpu"]["mean_w"], 3),
            "gpu_w": round(idle["gpu"]["mean_w"], 3),
            "cpu_samples": idle["cpu"]["samples"],
            "gpu_samples": idle["gpu"]["samples"],
        },
        flush=True,
    )
    pipeline(namespace, model, warm_moving, warm_fixed, device)

    rows = []
    for pair_index, (moving_idx, fixed_idx) in enumerate(pairs):
        moving_raw, fixed_raw = load_raw_pair(
            namespace, dataset, moving_idx, fixed_idx
        )
        latency_runs = []
        last_warped = None
        for _ in range(LATENCY_REPETITIONS):
            timing, last_warped = pipeline(
                namespace, model, moving_raw, fixed_raw, device
            )
            latency_runs.append(timing)
        if last_warped.shape != moving_raw.shape:
            raise RuntimeError(
                f"Unexpected warped shape {last_warped.shape}; "
                f"expected {moving_raw.shape}"
            )
        latency_mean_ms, latency_sd_ms = _mean_sd(
            run["total_runtime_ms"] for run in latency_runs
        )
        model_mean_ms, model_sd_ms = _mean_sd(
            run["model_inference_ms"] for run in latency_runs
        )
        warp_mean_ms, warp_sd_ms = _mean_sd(
            run["warp_ms"] for run in latency_runs
        )
        with SampledPowerMonitor(
            POWER_SAMPLE_INTERVAL_SECONDS
        ) as monitor:
            repetitions = 0
            power_started = time.perf_counter()
            while (
                repetitions < 2
                or time.perf_counter() - power_started
                < POWER_WINDOW_SECONDS
            ):
                _, last_warped = pipeline(
                    namespace, model, moving_raw, fixed_raw, device
                )
                repetitions += 1
        power = monitor.result()
        if power["cpu"]["mean_w"] is None:
            raise RuntimeError("CPU power sampling failed during benchmark")
        if power["gpu"]["mean_w"] is None:
            raise RuntimeError("GPU power sampling failed during benchmark")

        row = {
            "configuration": config_name,
            "model": "2.5d" if "2.5d" in config_name else "3d",
            "device": device.type,
            "pair_index": pair_index,
            "moving_idx": int(moving_idx),
            "fixed_idx": int(fixed_idx),
            "latency_repetitions": LATENCY_REPETITIONS,
            "latency_ms": latency_mean_ms,
            "latency_sd_within_pair_ms": latency_sd_ms,
            "model_inference_ms": model_mean_ms,
            "model_inference_sd_within_pair_ms": model_sd_ms,
            "warp_ms": warp_mean_ms,
            "warp_sd_within_pair_ms": warp_sd_ms,
            "idle_cpu_power_w": idle["cpu"]["mean_w"],
            "idle_gpu_power_w": idle["gpu"]["mean_w"],
            **_power_fields(power, idle, repetitions),
        }
        rows.append(row)
        print(
            f"  Pair {pair_index + 1}/{len(pairs)}: "
            f"latency={latency_mean_ms:.2f} ms, "
            f"window={row['power_window_s']:.2f}s/"
            f"{repetitions}, "
            f"power={row['power_mean_w']:.2f} W, "
            f"energy={row['energy_j_per_inference']:.3f} J, "
            f"dynamic={row['dynamic_energy_j_per_inference']:.3f} J",
            flush=True,
        )
        del last_warped
        gc.collect()

    return rows, idle


def aggregate(rows: list[dict], configuration: str) -> dict:
    selected = [
        row for row in rows if row["configuration"] == configuration
    ]
    keys = (
        "latency_ms",
        "model_inference_ms",
        "warp_ms",
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
    result = {"pair_count": len(selected)}
    for key in keys:
        mean, sd = _mean_sd(row[key] for row in selected)
        result[f"{key}_mean"] = mean
        result[f"{key}_sd"] = sd
    result["idle_cpu_power_w"] = float(selected[0]["idle_cpu_power_w"])
    result["idle_gpu_power_w"] = float(selected[0]["idle_gpu_power_w"])
    return result


def main() -> None:
    os.environ["V4_MEASURE_LOCAL_POWER"] = "0"
    namespace = load_measurement_namespace()
    dataset = namespace["load_split"]("test", None)
    pairs = namespace["build_eval_pairs"](
        len(dataset), max_pairs=None, seed=namespace["PAIR_SEED"]
    )
    if PAIR_LIMIT > 0:
        pairs = pairs[:PAIR_LIMIT]
    if PAIR_LIMIT <= 0 and len(pairs) != 9:
        raise RuntimeError(f"Expected nine pairs, received {len(pairs)}")

    rows: list[dict] = []
    idle_calibrations = {}
    configurations = []
    cpu = torch.device("cpu")
    cpu_models = (
        ("2.5d_cpu", run_v4_pipeline, _load_v4_model(namespace, cpu)),
        ("3d_cpu", run_3d_pipeline, _load_3d_model(namespace, cpu)),
    )
    for name, pipeline, model in cpu_models:
        config_rows, idle = benchmark_configuration(
            namespace,
            dataset,
            pairs,
            name,
            pipeline,
            model,
            cpu,
        )
        rows.extend(config_rows)
        idle_calibrations[name] = idle
        configurations.append(name)
        del model
        gc.collect()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    cuda = torch.device("cuda")
    gpu_models = (
        ("2.5d_gpu", run_v4_pipeline, _load_v4_model(namespace, cuda)),
        ("3d_gpu", run_3d_pipeline, _load_3d_model(namespace, cuda)),
    )
    for name, pipeline, model in gpu_models:
        config_rows, idle = benchmark_configuration(
            namespace,
            dataset,
            pairs,
            name,
            pipeline,
            model,
            cuda,
        )
        rows.extend(config_rows)
        idle_calibrations[name] = idle
        configurations.append(name)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    models = {
        name: aggregate(rows, name) for name in configurations
    }
    payload = {
        "schema_version": 1,
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "pair_count": len(pairs),
        "settings": {
            "latency_repetitions": LATENCY_REPETITIONS,
            "minimum_power_window_s": POWER_WINDOW_SECONDS,
            "power_sample_interval_s": POWER_SAMPLE_INTERVAL_SECONDS,
            "idle_calibration_s": IDLE_CALIBRATION_SECONDS,
        },
        "scope": {
            "start": (
                "normalized raw in-memory moving/fixed volumes enter "
                "model preparation"
            ),
            "end": (
                "warped moving intensity volume is available in host memory"
            ),
            "included": (
                "model-input preparation, transfers, model execution, "
                "three-orientation fusion or 3D flow upsampling, flow resizing, "
                "and intensity-volume warping"
            ),
            "excluded": (
                "disk I/O, segmentation warping, Dice, TRE, MI, SSIM, "
                "plotting, and report generation"
            ),
        },
        "sensor_coverage": {
            "cpu": (
                "AMD CPU PPT/package power from LibreHardwareMonitor web API"
            ),
            "gpu": "NVIDIA GPU board power from nvidia-smi",
            "total": (
                "sum of CPU package/PPT and NVIDIA GPU board power; excludes "
                "RAM, motherboard, storage, display, PSU losses, and other rails"
            ),
            "dynamic_energy": (
                "per-configuration idle-subtracted CPU plus GPU energy, "
                "clamped independently at zero"
            ),
        },
        "idle_calibrations": idle_calibrations,
        "models": models,
        "pairs": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("\nNine-pair aggregates:", flush=True)
    print(json.dumps(models, indent=2), flush=True)
    print(f"Wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
