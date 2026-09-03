#!/usr/bin/env python3
"""Run WSL TensorRT INT8 while sampling Windows CPU PPT and GPU board power."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import threading
import os
import time
from pathlib import Path

import numpy as np

from run_windows_benchmark import (
    query_cpu_power_w,
    query_gpu_power_w,
    select_cpu_power_sensor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "Voxelmorph"
    / "artifacts"
    / "results"
    / "inference_only_local_power_latency"
    / "v4_gpu_int8_tensorrt.json"
)
WSL_ENGINE = (
    "/home/j/FPGA_Inference/out/v4_canonical/"
    "vxm_2p5d_pt_v4_canonical/host_exports/"
    "Vxm2p5dV4_int8_tensorrt.plan"
)
WSL_DATA_ROOT = os.environ.get(
    "REGISTRATION_DATA_ROOT_WSL",
    "/mnt/d/FPGA_Inference/Data/registration_dataset",
)
WSL_PYTHON = "/home/j/miniconda3/envs/gen/bin/python"
LD_LIBRARY_PATH = (
    "/home/j/miniconda3/envs/gen/lib/python3.10/site-packages/nvidia/cublas/lib:"
    "/home/j/miniconda3/envs/gen/lib/python3.10/site-packages/nvidia/cu13/lib:"
    "/home/j/miniconda3/envs/gen/lib"
)


class PowerMonitor:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.cpu_samples = []
        self.gpu_samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.started = None
        self.ended = None

    def sample(self) -> None:
        cpu = query_cpu_power_w()
        gpu = query_gpu_power_w()
        if cpu is not None:
            self.cpu_samples.append(float(cpu))
        if gpu is not None:
            self.gpu_samples.append(float(gpu))

    def loop(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self.sample()

    def __enter__(self):
        self.sample()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ended = time.perf_counter()
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
        self.sample()
        return False

    @staticmethod
    def summary(values) -> dict:
        array = np.asarray(values, dtype=np.float64)
        if not array.size:
            return {"mean_w": None, "sd_w": None, "peak_w": None, "samples": 0}
        return {
            "mean_w": float(array.mean()),
            "sd_w": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "peak_w": float(array.max()),
            "samples": int(array.size),
        }

    def result(self) -> dict:
        return {
            "wall_time_s": float(self.ended - self.started),
            "cpu": self.summary(self.cpu_samples),
            "gpu": self.summary(self.gpu_samples),
        }


def mean_sd(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if array.size > 1 else 0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--idle-seconds", type=float, default=8.0)
    parser.add_argument("--power-window-seconds", type=float, default=10.0)
    parser.add_argument("--latency-repetitions", type=int, default=3)
    return parser.parse_args()


def add_power_fields(
    row: dict,
    power: dict,
    idle: dict,
    repetitions: int,
) -> None:
    wall = float(power["wall_time_s"])
    cpu_mean = float(power["cpu"]["mean_w"])
    gpu_mean = float(power["gpu"]["mean_w"])
    idle_cpu = float(idle["cpu"]["mean_w"])
    idle_gpu = float(idle["gpu"]["mean_w"])
    row.update(
        {
            "idle_cpu_power_w": idle_cpu,
            "idle_gpu_power_w": idle_gpu,
            "power_window_s": wall,
            "power_repetitions": repetitions,
            "cpu_power_mean_w": cpu_mean,
            "cpu_power_sd_w": power["cpu"]["sd_w"],
            "cpu_power_peak_w": power["cpu"]["peak_w"],
            "cpu_power_samples": power["cpu"]["samples"],
            "gpu_power_mean_w": gpu_mean,
            "gpu_power_sd_w": power["gpu"]["sd_w"],
            "gpu_power_peak_w": power["gpu"]["peak_w"],
            "gpu_power_samples": power["gpu"]["samples"],
            "power_mean_w": cpu_mean + gpu_mean,
            "cpu_energy_j_per_inference": cpu_mean * wall / repetitions,
            "gpu_energy_j_per_inference": gpu_mean * wall / repetitions,
            "energy_j_per_inference": (
                (cpu_mean + gpu_mean) * wall / repetitions
            ),
            "cpu_dynamic_energy_j_per_inference": (
                max(cpu_mean - idle_cpu, 0.0) * wall / repetitions
            ),
            "gpu_dynamic_energy_j_per_inference": (
                max(gpu_mean - idle_gpu, 0.0) * wall / repetitions
            ),
            "dynamic_energy_j_per_inference": (
                (
                    max(cpu_mean - idle_cpu, 0.0)
                    + max(gpu_mean - idle_gpu, 0.0)
                )
                * wall
                / repetitions
            ),
        }
    )


def aggregate(rows: list[dict]) -> dict:
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
    )
    result = {"pair_count": len(rows)}
    for key in keys:
        mean, sd = mean_sd([row[key] for row in rows])
        result[f"{key}_mean"] = mean
        result[f"{key}_sd"] = sd
    result["idle_cpu_power_w"] = rows[0]["idle_cpu_power_w"]
    result["idle_gpu_power_w"] = rows[0]["idle_gpu_power_w"]
    latency_s = result["latency_ms_mean"] / 1000.0
    throughput = 1.0 / latency_s
    result["throughput_inferences_per_s"] = throughput
    result["performance_per_watt_inferences_per_s_per_w"] = (
        throughput / result["power_mean_w_mean"]
    )
    result["raw_energy_from_mean_power_j"] = (
        result["power_mean_w_mean"] * latency_s
    )
    return result


def main() -> None:
    args = parse_args()
    selected_sensor = select_cpu_power_sensor()
    if selected_sensor is None:
        raise RuntimeError("CPU PPT/package power sensor is unavailable")

    command = [
        "wsl.exe",
        "-d",
        "Ubuntu",
        "--cd",
        "/home/j/FPGA_Inference",
        "env",
        f"LD_LIBRARY_PATH={LD_LIBRARY_PATH}",
        f"REGISTRATION_DATA_ROOT={WSL_DATA_ROOT}",
        "V4_MEASURE_LOCAL_POWER=0",
        "PYTHONUNBUFFERED=1",
        WSL_PYTHON,
        "-u",
        "Voxelmorph/benchmark_v4_tensorrt_int8_child.py",
        "--engine",
        WSL_ENGINE,
        "--pair-limit",
        str(args.pair_limit),
        "--latency-repetitions",
        str(args.latency_repetitions),
        "--power-window-seconds",
        str(args.power_window_seconds),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("Could not open child process pipes")

    idle = None
    rows = []
    complete = False
    while True:
        line = process.stdout.readline()
        if not line:
            break
        text = line.rstrip()
        if text.startswith("POWER_READY "):
            ready = json.loads(text[len("POWER_READY ") :])
            if idle is None:
                print(
                    f"Idle calibration ({args.idle_seconds:.1f}s) with engine loaded",
                    flush=True,
                )
                with PowerMonitor(args.sample_interval) as monitor:
                    time.sleep(args.idle_seconds)
                idle = monitor.result()
                if idle["cpu"]["mean_w"] is None or idle["gpu"]["mean_w"] is None:
                    raise RuntimeError("Idle power sampling failed")
            with PowerMonitor(args.sample_interval) as monitor:
                process.stdin.write("START\n")
                process.stdin.flush()
                done_line = process.stdout.readline().rstrip()
                while not done_line.startswith("POWER_DONE "):
                    print(done_line, flush=True)
                    done_line = process.stdout.readline().rstrip()
                    if not done_line and process.poll() is not None:
                        raise RuntimeError("TensorRT child exited during power run")
            done = json.loads(done_line[len("POWER_DONE ") :])
            power = monitor.result()
            if power["cpu"]["mean_w"] is None or power["gpu"]["mean_w"] is None:
                raise RuntimeError("Workload power sampling failed")
            row = {
                "configuration": "gpu_int8",
                "model": "2.5d",
                "device": "cuda",
                **ready,
                "child_power_wall_time_s": done["child_power_wall_time_s"],
            }
            add_power_fields(row, power, idle, int(done["power_repetitions"]))
            rows.append(row)
            print(
                f"Pair {len(rows)}: latency={row['latency_ms']:.2f} ms, "
                f"power={row['power_mean_w']:.2f} W",
                flush=True,
            )
        elif text == "CHILD_COMPLETE":
            complete = True
        else:
            print(text, flush=True)

    return_code = process.wait()
    if return_code != 0 or not complete:
        raise RuntimeError(
            f"TensorRT child failed with exit code {return_code}"
        )
    expected = args.pair_limit if args.pair_limit > 0 else 9
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, received {len(rows)}")

    model = aggregate(rows)
    payload = {
        "schema_version": 1,
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "platform": platform.platform(),
        "pair_count": len(rows),
        "timing_scope": (
            "normalized in-memory volumes through three-orientation model "
            "execution, mean fusion, flow resizing, and warped intensity output"
        ),
        "power_scope": {
            "cpu": "AMD CPU PPT/package power",
            "gpu": "NVIDIA GPU board power",
            "total": "CPU PPT/package plus NVIDIA GPU board power",
            "power_mean_w": "raw mean during workload; idle is not subtracted",
        },
        "sensor_selection": {
            "cpu": selected_sensor,
            "gpu": {
                "name": "NVIDIA GPU board power.draw",
                "backend": "nvidia-smi",
            },
        },
        "quantization": {
            "precision": "INT8",
            "source": "Vitis AI 2.5 NNDCT Q/DQ ONNX",
            "backend": "TensorRT 8.5.3.1",
            "audit": (
                "all 17 convolutions use INT8 weights and INT8 TensorRT tactics; "
                "TF32 disabled; no FP16 layers"
            ),
            "host_batch": "dynamic batch 1-16; existing host benchmark uses 16",
        },
        "idle_calibration": idle,
        "models": {"gpu_int8": model},
        "pairs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["models"], indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
