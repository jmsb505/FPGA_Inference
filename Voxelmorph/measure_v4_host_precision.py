#!/usr/bin/env python3
"""Measure V4 FP32 and native INT8 host deployments on the matched pipeline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import measure_local_inference_power_latency as benchmark
from measure_3d_gpu_power import load_measurement_namespace


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INT8_ONNX = (
    REPO_ROOT
    / "out"
    / "v4_canonical"
    / "vxm_2p5d_pt_v4_canonical"
    / "host_exports"
    / "Vxm2p5dV4_ort_qoperator_int8.onnx"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "Voxelmorph"
    / "artifacts"
    / "results"
    / "inference_only_local_power_latency"
    / "v4_fp32_int8_host_comparison.json"
)


class OrtCpuModule:
    def __init__(self, model_path: Path) -> None:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            options,
            providers=["CPUExecutionProvider"],
        )
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"Unexpected ONNX Runtime providers: {self.session.get_providers()}"
            )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        output = self.session.run(None, {self.input_name: array})[0]
        return torch.from_numpy(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configurations",
        default="cpu_fp32,cpu_int8,gpu_fp32",
        help="Comma-separated subset of cpu_fp32,cpu_int8,gpu_fp32",
    )
    parser.add_argument("--int8-onnx", type=Path, default=DEFAULT_INT8_ONNX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def enrich_aggregate(values: dict) -> dict:
    latency_s = float(values["latency_ms_mean"]) / 1000.0
    power_w = float(values["power_mean_w_mean"])
    throughput = 1.0 / latency_s
    values["throughput_inferences_per_s"] = throughput
    values["performance_per_watt_inferences_per_s_per_w"] = throughput / power_w
    values["raw_energy_from_mean_power_j"] = power_w * latency_s
    return values


def main() -> None:
    args = parse_args()
    requested = [
        item.strip() for item in args.configurations.split(",") if item.strip()
    ]
    allowed = {"cpu_fp32", "cpu_int8", "gpu_fp32"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"Unknown configurations: {unknown}")

    os.environ["V4_MEASURE_LOCAL_POWER"] = "0"
    namespace = load_measurement_namespace()
    dataset = namespace["load_split"]("test", None)
    pairs = namespace["build_eval_pairs"](
        len(dataset), max_pairs=None, seed=namespace["PAIR_SEED"]
    )
    if benchmark.PAIR_LIMIT > 0:
        pairs = pairs[: benchmark.PAIR_LIMIT]
    elif len(pairs) != 9:
        raise RuntimeError(f"Expected nine pairs, received {len(pairs)}")

    selected_sensor = benchmark.SELECT_CPU_POWER_SENSOR()
    if selected_sensor is None:
        raise RuntimeError("CPU PPT/package power sensor is unavailable")
    rows = []
    idle_calibrations = {}
    completed = []

    for name in requested:
        if name == "cpu_fp32":
            device = torch.device("cpu")
            model = benchmark._load_v4_model(namespace, device)
        elif name == "cpu_int8":
            device = torch.device("cpu")
            model = OrtCpuModule(args.int8_onnx.resolve())
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            device = torch.device("cuda")
            model = benchmark._load_v4_model(namespace, device)

        config_rows, idle = benchmark.benchmark_configuration(
            namespace,
            dataset,
            pairs,
            name,
            benchmark.run_v4_pipeline,
            model,
            device,
        )
        rows.extend(config_rows)
        idle_calibrations[name] = idle
        completed.append(name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregates = {
        name: enrich_aggregate(benchmark.aggregate(rows, name))
        for name in completed
    }
    payload = {
        "schema_version": 1,
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "pair_count": len(pairs),
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
                "device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            },
        },
        "quantization": {
            "cpu_int8": (
                "ONNX Runtime static QOperator PTQ, 64 existing V4 calibration "
                "samples, 17 QLinearConv and 15 QLinearLeakyRelu operators"
            ),
            "cpu_fp32": "PyTorch FP32",
            "gpu_fp32": "PyTorch FP32",
        },
        "settings": {
            "latency_repetitions": benchmark.LATENCY_REPETITIONS,
            "minimum_power_window_s": benchmark.POWER_WINDOW_SECONDS,
            "power_sample_interval_s": benchmark.POWER_SAMPLE_INTERVAL_SECONDS,
            "idle_calibration_s": benchmark.IDLE_CALIBRATION_SECONDS,
        },
        "idle_calibrations": idle_calibrations,
        "models": aggregates,
        "pairs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregates, indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
