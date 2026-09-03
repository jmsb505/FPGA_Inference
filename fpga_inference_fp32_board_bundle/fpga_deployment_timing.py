"""Metric-free V4 deployment timing for FPGA DPU and ARM CPU."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np


SUBJECTS = [
    "1BA111",
    "1BA116",
    "1BA125",
    "1BA131",
    "1BA141",
    "1BA143",
    "1BA151",
    "1BA158",
    "1BA159",
    "1BA164",
    "1BA172",
    "1BA175",
    "1BA184",
    "1BA185",
    "1BA189",
    "1BA201",
    "1BA206",
    "1BA220",
]
PAIRS = [
    (2, 17),
    (1, 12),
    (6, 0),
    (16, 4),
    (14, 8),
    (5, 9),
    (13, 11),
    (7, 10),
    (15, 3),
]


def _load_volume(namespace: dict, subject: str, modality: str) -> np.ndarray:
    resolve_file = namespace["resolve_file"]
    path = resolve_file(
        f"./{subject}_{modality}.npy",
        [f"./data/test_data/{subject}_{modality}.npy"],
    )
    return namespace["normalize_volume_contract"](
        np.load(path).astype(np.float32)
    )


def _prepare_canonical(
    namespace: dict,
    moving_raw: np.ndarray,
    fixed_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    moving = namespace["normalize_volume_contract"](
        namespace["resample_volume_v4"](moving_raw)
    )
    fixed = namespace["normalize_volume_contract"](
        namespace["resample_volume_v4"](fixed_raw)
    )
    return moving, fixed


def _warm_device(
    namespace: dict,
    moving_raw: np.ndarray,
    fixed_raw: np.ndarray,
    device: str,
    weights_path: str,
) -> None:
    moving, fixed = _prepare_canonical(namespace, moving_raw, fixed_raw)
    z = moving.shape[0] // 2
    moving_stack = namespace["letterbox_stack_v4"](
        namespace["extract_v4_stack"](moving, "axial", z), "axial"
    )
    fixed_stack = namespace["letterbox_stack_v4"](
        namespace["extract_v4_stack"](fixed, "axial", z), "axial"
    )
    if device == "fpga":
        namespace["run_dpu_inference"](moving_stack, fixed_stack)
    else:
        namespace["run_cpu_inference"](
            moving_stack, fixed_stack, weights_path
        )


def _run_dpu_stack(
    namespace: dict,
    moving_stack: np.ndarray,
    fixed_stack: np.ndarray,
) -> tuple[np.ndarray, float]:
    combined = np.concatenate([moving_stack, fixed_stack], axis=0)
    input_data = combined.transpose(1, 2, 0)[None].astype(np.float32)
    quantized = np.clip(
        np.round(input_data * namespace["INPUT_SCALE"]), -128, 127
    ).astype(np.int8)
    input_buffer = output_buffer = None
    try:
        input_buffer = namespace["allocate"](
            shape=tuple(input_data.shape), dtype=np.int8
        )
        np.copyto(input_buffer, np.ascontiguousarray(quantized))
        if hasattr(input_buffer, "sync_to_device"):
            input_buffer.sync_to_device()
        output_buffer = namespace["allocate"](
            shape=tuple(namespace["output_shape"]), dtype=np.int8
        )
        started = time.perf_counter()
        job_id = namespace["dpu_runner"].execute_async(
            [input_buffer], [output_buffer]
        )
        namespace["dpu_runner"].wait(job_id)
        dpu_seconds = time.perf_counter() - started
        if hasattr(output_buffer, "sync_from_device"):
            output_buffer.sync_from_device()
        output = np.array(output_buffer, copy=True)
    finally:
        if input_buffer is not None and hasattr(input_buffer, "freebuffer"):
            input_buffer.freebuffer()
        if output_buffer is not None and hasattr(output_buffer, "freebuffer"):
            output_buffer.freebuffer()
    flow = output.astype(np.float32) / namespace["OUTPUT_SCALE"]
    return np.ascontiguousarray(flow[0].transpose(2, 0, 1)), dpu_seconds


def _infer_orientation(
    namespace: dict,
    moving: np.ndarray,
    fixed: np.ndarray,
    orientation: str,
    device: str,
    weights_path: str,
) -> tuple[np.ndarray, list[float]]:
    axis = {"axial": 0, "coronal": 1, "sagittal": 2}[orientation]
    flows = []
    durations = []
    for z in range(
        namespace["WINDOW_RADIUS"],
        moving.shape[axis] - namespace["WINDOW_RADIUS"],
    ):
        moving_stack = namespace["letterbox_stack_v4"](
            namespace["extract_v4_stack"](moving, orientation, z),
            orientation,
        )
        fixed_stack = namespace["letterbox_stack_v4"](
            namespace["extract_v4_stack"](fixed, orientation, z),
            orientation,
        )
        if device == "fpga":
            flow, elapsed = _run_dpu_stack(
                namespace, moving_stack, fixed_stack
            )
        else:
            flow, elapsed = namespace["run_cpu_inference"](
                moving_stack, fixed_stack, weights_path
            )
        flows.append(flow)
        durations.append(float(elapsed))
    stack = np.stack(flows).astype(np.float32)
    return (
        namespace["canvas_flow_to_native_v4"](stack, orientation),
        durations,
    )

def _measure_pair(
    namespace: dict,
    moving_raw: np.ndarray,
    fixed_raw: np.ndarray,
    device: str,
    weights_path: str,
) -> tuple[dict, np.ndarray]:
    started = time.perf_counter()
    moving, fixed = _prepare_canonical(namespace, moving_raw, fixed_raw)

    flows = {}
    model_seconds = 0.0
    for orientation in ("axial", "coronal", "sagittal"):
        flow, durations = _infer_orientation(
            namespace,
            moving,
            fixed,
            orientation,
            device,
            weights_path,
        )
        flows[orientation] = flow
        model_seconds += float(np.sum(durations))

    axial = namespace["lift_axial_v4"](flows["axial"])
    coronal = namespace["lift_coronal_v4"](flows["coronal"])
    sagittal = namespace["lift_sagittal_v4"](flows["sagittal"])
    fused = namespace["fuse_fields"](axial, coronal, sagittal)
    raw_flow = namespace["canonical_field_to_raw"](fused, moving_raw.shape)
    warped = namespace["warp_volume_3d_numpy"](
        moving_raw, raw_flow, mode="linear"
    )
    total_ms = (time.perf_counter() - started) * 1000.0
    model_ms = model_seconds * 1000.0

    if device == "fpga":
        accelerator_ms = model_ms
        cpu_host_ms = max(total_ms - accelerator_ms, 0.0)
        cpu_model_ms = 0.0
    else:
        accelerator_ms = 0.0
        cpu_host_ms = total_ms
        cpu_model_ms = model_ms

    return (
        {
            "model": (
                "2.5d_fused_dpu"
                if device == "fpga"
                else "2.5d_fused_arm_cpu"
            ),
            "accelerator": "dpu" if device == "fpga" else "none",
            "model_inference_ms": model_ms,
            "accelerator_warp_ms": 0.0,
            "accelerator_ms": accelerator_ms,
            "cpu_model_ms": cpu_model_ms,
            "cpu_host_ms": cpu_host_ms,
            "cpu_non_model_ms": max(cpu_host_ms - cpu_model_ms, 0.0),
            "total_runtime_ms": total_ms,
        },
        warped,
    )


def _aggregate(rows: list[dict], model: str) -> dict:
    selected = [row for row in rows if row["model"] == model]
    keys = (
        "model_inference_ms",
        "accelerator_warp_ms",
        "accelerator_ms",
        "cpu_model_ms",
        "cpu_host_ms",
        "cpu_non_model_ms",
        "total_runtime_ms",
    )
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in selected], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_sd"] = (
            float(values.std(ddof=1)) if values.size > 1 else 0.0
        )
    result["pair_count"] = len(selected)
    return result


def run_benchmark(
    namespace: dict,
    output_path: str = "inference_pipeline_breakdown_v4.json",
) -> dict:
    weights_path = namespace["resolve_file"](
        "./2p5d_dense_pt_v4_canonical_best.pth",
        [
            "../trained_weights/2p5d_dense_pt_v4_canonical_best.pth",
            (
                "/home/xilinx/jupyter_notebooks/fpga_run_new/"
                "2p5d_dense_pt_v4_canonical_best.pth"
            ),
        ],
    )
    warm_moving = _load_volume(namespace, SUBJECTS[PAIRS[0][0]], "mr")
    warm_fixed = _load_volume(namespace, SUBJECTS[PAIRS[0][1]], "ct")
    for device in ("fpga", "cpu"):
        print(f"Warming {device.upper()} timing path...")
        _warm_device(
            namespace, warm_moving, warm_fixed, device, weights_path
        )

    rows = []
    for device in ("fpga", "cpu"):
        print(f"\nMetric-free deployment timing: {device.upper()}")
        for pair_index, (moving_idx, fixed_idx) in enumerate(PAIRS):
            moving_id = SUBJECTS[moving_idx]
            fixed_id = SUBJECTS[fixed_idx]
            moving_raw = _load_volume(namespace, moving_id, "mr")
            fixed_raw = _load_volume(namespace, fixed_id, "ct")
            result, warped = _measure_pair(
                namespace,
                moving_raw,
                fixed_raw,
                device,
                weights_path,
            )
            if warped.shape != moving_raw.shape:
                raise RuntimeError(
                    f"Unexpected warped shape {warped.shape}; "
                    f"expected {moving_raw.shape}"
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
                f"CPU={result['cpu_host_ms']:.1f} ms, "
                f"total={result['total_runtime_ms']:.1f} ms"
            )
            del warped
            gc.collect()

    model_names = ("2.5d_fused_dpu", "2.5d_fused_arm_cpu")
    models = {model: _aggregate(rows, model) for model in model_names}
    payload = {
        "schema_version": 1,
        "scope": {
            "start": (
                "normalized raw in-memory moving/fixed volumes enter "
                "model preparation"
            ),
            "end": "warped moving intensity volume is available in host memory",
            "included": (
                "model-input preparation, transfers/quantization, model "
                "execution, three-orientation lifting/fusion, flow resizing, "
                "and intensity-volume warping"
            ),
            "excluded": (
                "disk I/O, segmentation warping, Dice, TRE, MI, SSIM, "
                "power sampling, plotting, and report generation"
            ),
            "cpu_definition": (
                "DPU: total wall minus DPU runner time; ARM CPU: total wall"
            ),
            "accelerator_definition": (
                "DPU execute_async/wait only; the ARM CPU path has no "
                "accelerator component"
            ),
        },
        "pair_count": len(PAIRS),
        "models": models,
        "pairs": rows,
    }
    destination = Path(output_path)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nNine-pair averages:")
    print(json.dumps(models, indent=2))
    print(f"Wrote {destination.resolve()}")
    return payload
