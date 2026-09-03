"""Native ARM CPU INT8 benchmark using the existing board measurement protocol."""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import fpga_deployment_power as board_power
from fpga_deployment_timing import PAIRS, SUBJECTS, _load_volume


def _ort_cpu_runner(model_path):
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(
            f"Unexpected ONNX Runtime providers: {session.get_providers()}"
        )
    input_name = session.get_inputs()[0].name

    def run_cpu_int8(moving_stack, fixed_stack, _weights_path):
        combined = np.concatenate([moving_stack, fixed_stack], axis=0)
        input_data = combined.transpose(1, 2, 0)[None].astype(np.float32)
        started = time.perf_counter()
        output = session.run(None, {input_name: input_data})[0]
        elapsed_s = time.perf_counter() - started
        return np.ascontiguousarray(output[0]), elapsed_s

    return session, run_cpu_int8


def run_arm_int8_benchmark(
    namespace,
    model_path="Vxm2p5dV4_ort_qoperator_int8.onnx",
    output_path="inference_pipeline_power_latency_v4_arm_int8.json",
    latency_repetitions=3,
    minimum_power_window_s=10.0,
    power_sample_interval_s=0.10,
    idle_calibration_s=20.0,
):
    """Run after the setup and utility cells in the board V4 notebook."""
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ARM INT8 ONNX model: {model_path}")

    session, runner = _ort_cpu_runner(model_path)
    runtime_namespace = dict(namespace)
    runtime_namespace["run_cpu_inference"] = runner
    weights_path = board_power._resolve_weights(runtime_namespace)

    warm_moving = _load_volume(
        runtime_namespace, SUBJECTS[PAIRS[0][0]], "mr"
    )
    warm_fixed = _load_volume(
        runtime_namespace, SUBJECTS[PAIRS[0][1]], "ct"
    )
    board_power._warm_complete_pipeline(
        runtime_namespace,
        warm_moving,
        warm_fixed,
        "cpu",
        weights_path,
    )
    idle = board_power._calibrate_idle(
        runtime_namespace,
        duration_s=idle_calibration_s,
        sample_interval_s=power_sample_interval_s,
    )
    idle_cpu_w = float(idle["arm_cpu"]["mean_w"])
    idle_dpu_w = float(idle["dpu"]["mean_w"])

    rows = []
    for pair_index, (moving_idx, fixed_idx) in enumerate(PAIRS):
        moving_raw = _load_volume(
            runtime_namespace, SUBJECTS[moving_idx], "mr"
        )
        fixed_raw = _load_volume(
            runtime_namespace, SUBJECTS[fixed_idx], "ct"
        )
        result = board_power._measure_latency(
            runtime_namespace,
            moving_raw,
            fixed_raw,
            "cpu",
            weights_path,
            repetitions=latency_repetitions,
        )
        result.update(
            board_power._measure_power(
                runtime_namespace,
                moving_raw,
                fixed_raw,
                "cpu",
                weights_path,
                idle_cpu_w=idle_cpu_w,
                idle_dpu_w=idle_dpu_w,
                minimum_window_s=minimum_power_window_s,
                sample_interval_s=power_sample_interval_s,
            )
        )
        result.update(
            model="2.5d_fused_arm_cpu_int8",
            precision="INT8",
            backend="ONNX Runtime CPUExecutionProvider QOperator",
            pair_index=pair_index,
            moving_idx=moving_idx,
            fixed_idx=fixed_idx,
        )
        rows.append(result)
        print(
            f"Pair {pair_index + 1}/{len(PAIRS)}: "
            f"latency={result['total_runtime_ms']:.1f} ms, "
            f"power={result['power_mean_w']:.3f} W"
        )
        gc.collect()

    model_name = "2.5d_fused_arm_cpu_int8"
    model = board_power._aggregate(rows, model_name)
    latency_s = model["total_runtime_ms_mean"] / 1000.0
    throughput = 1.0 / latency_s
    model["throughput_inferences_per_s"] = throughput
    model["performance_per_watt_inferences_per_s_per_w"] = (
        throughput / model["power_mean_w_mean"]
    )
    model["raw_energy_from_mean_power_j"] = (
        model["power_mean_w_mean"] * latency_s
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "pair_count": len(rows),
        "model_path": str(model_path),
        "quantization": {
            "precision": "INT8",
            "backend": "ONNX Runtime CPUExecutionProvider QOperator",
            "calibration": "64 existing V4 calibration samples",
            "operator_audit": "17 QLinearConv and 15 QLinearLeakyRelu",
        },
        "scope": {
            "timing": (
                "normalized in-memory volumes through three-orientation model "
                "execution, mean fusion, flow resizing, and warped intensity"
            ),
            "power": (
                "PSINTFP ARM/PS plus INT PL rail raw mean; not whole-board power"
            ),
        },
        "idle_calibration": idle,
        "models": {model_name: model},
        "pairs": rows,
    }
    destination = Path(output_path)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["models"], indent=2))
    print(f"Wrote {destination.resolve()}")
    del session
    return payload


if __name__ == "__main__":
    raise SystemExit(
        "Run this module from the board V4 notebook after its setup and "
        "utility cells, then call run_arm_int8_benchmark(globals())."
    )
