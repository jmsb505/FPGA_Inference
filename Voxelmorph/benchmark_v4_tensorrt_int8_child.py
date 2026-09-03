#!/usr/bin/env python3
"""Child protocol for matched TensorRT INT8 latency and power workloads."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

import measure_local_inference_power_latency as benchmark
from measure_3d_gpu_power import load_measurement_namespace
from measure_inference_pipeline_breakdown import load_raw_pair


def mean_sd(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if array.size > 1 else 0.0,
    )


class TensorRTModule:
    def __init__(self, engine_path: Path) -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            engine_path.resolve().read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        self.input_index = next(
            index
            for index in range(self.engine.num_bindings)
            if self.engine.binding_is_input(index)
        )
        self.output_index = next(
            index
            for index in range(self.engine.num_bindings)
            if not self.engine.binding_is_input(index)
        )

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        value = value.contiguous().to(dtype=torch.float32, device="cuda")
        if not self.context.set_binding_shape(
            self.input_index, tuple(value.shape)
        ):
            raise RuntimeError(f"TensorRT rejected input shape {tuple(value.shape)}")
        output_shape = tuple(self.context.get_binding_shape(self.output_index))
        output = torch.empty(output_shape, dtype=torch.float32, device="cuda")
        bindings = [0] * self.engine.num_bindings
        bindings[self.input_index] = int(value.data_ptr())
        bindings[self.output_index] = int(output.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v2(bindings, stream):
            raise RuntimeError("TensorRT execute_async_v2 failed")
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument("--latency-repetitions", type=int, default=3)
    parser.add_argument("--power-window-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["V4_MEASURE_LOCAL_POWER"] = "0"
    namespace = load_measurement_namespace()
    dataset = namespace["load_split"]("test", None)
    pairs = namespace["build_eval_pairs"](
        len(dataset), max_pairs=None, seed=namespace["PAIR_SEED"]
    )
    if args.pair_limit > 0:
        pairs = pairs[: args.pair_limit]
    elif len(pairs) != 9:
        raise RuntimeError(f"Expected nine pairs, received {len(pairs)}")

    device = torch.device("cuda")
    model = TensorRTModule(args.engine)
    moving_raw, fixed_raw = load_raw_pair(namespace, dataset, *pairs[0])
    benchmark.run_v4_pipeline(
        namespace, model, moving_raw, fixed_raw, device
    )
    torch.cuda.synchronize()
    gc.collect()

    for pair_index, (moving_idx, fixed_idx) in enumerate(pairs):
        moving_raw, fixed_raw = load_raw_pair(
            namespace, dataset, moving_idx, fixed_idx
        )
        runs = []
        for _ in range(args.latency_repetitions):
            timing, warped = benchmark.run_v4_pipeline(
                namespace, model, moving_raw, fixed_raw, device
            )
            if warped.shape != moving_raw.shape:
                raise RuntimeError(
                    f"Unexpected warped shape {warped.shape}; expected {moving_raw.shape}"
                )
            runs.append(timing)
        latency_mean, latency_sd = mean_sd(
            [run["total_runtime_ms"] for run in runs]
        )
        model_mean, model_sd = mean_sd(
            [run["model_inference_ms"] for run in runs]
        )
        warp_mean, warp_sd = mean_sd([run["warp_ms"] for run in runs])
        ready = {
            "pair_index": pair_index,
            "moving_idx": int(moving_idx),
            "fixed_idx": int(fixed_idx),
            "latency_repetitions": args.latency_repetitions,
            "latency_ms": latency_mean,
            "latency_sd_within_pair_ms": latency_sd,
            "model_inference_ms": model_mean,
            "model_inference_sd_within_pair_ms": model_sd,
            "warp_ms": warp_mean,
            "warp_sd_within_pair_ms": warp_sd,
        }
        print("POWER_READY " + json.dumps(ready), flush=True)
        if input().strip() != "START":
            raise RuntimeError("Parent did not send START")

        started = time.perf_counter()
        repetitions = 0
        while (
            repetitions < 2
            or time.perf_counter() - started < args.power_window_seconds
        ):
            benchmark.run_v4_pipeline(
                namespace, model, moving_raw, fixed_raw, device
            )
            repetitions += 1
        torch.cuda.synchronize()
        done = {
            "pair_index": pair_index,
            "power_repetitions": repetitions,
            "child_power_wall_time_s": time.perf_counter() - started,
        }
        print("POWER_DONE " + json.dumps(done), flush=True)

    print("CHILD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
