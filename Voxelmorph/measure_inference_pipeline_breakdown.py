"""Measure metric-free registration latency with CPU/accelerator separation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from measure_3d_gpu_power import load_measurement_namespace


REPO_ROOT = Path(__file__).resolve().parent.parent
VOX_DIR = REPO_ROOT / "Voxelmorph"
OUTPUT_PATH = (
    VOX_DIR
    / "artifacts"
    / "results"
    / "inference_pipeline_breakdown"
    / "local_gpu_9_pairs.json"
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_cuda(stage, device: torch.device):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = stage()
    end_event.record()
    synchronize(device)
    return result, float(start_event.elapsed_time(end_event))


def load_raw_pair(namespace: dict, dataset, moving_idx: int, fixed_idx: int):
    moving, _, _, _ = namespace["get_sample_parts"](dataset, moving_idx)
    _, fixed, _, _ = namespace["get_sample_parts"](dataset, fixed_idx)
    return np.asarray(moving, dtype=np.float32), np.asarray(fixed, dtype=np.float32)


@torch.no_grad()
def measure_v4_fused(
    namespace: dict,
    model,
    moving_raw: np.ndarray,
    fixed_raw: np.ndarray,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
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

    warped, warp_gpu_ms = timed_cuda(
        lambda: namespace["warp_volume_3d"](
            moving_raw,
            raw_flow,
            mode="bilinear",
            runtime_device=device,
        ),
        device,
    )
    total_ms = (time.perf_counter() - total_started) * 1000.0
    accelerator_ms = model_ms + warp_gpu_ms
    result = {
        "model": "2.5d_fused_gpu",
        "accelerator": "gpu",
        "model_inference_ms": model_ms,
        "accelerator_warp_ms": warp_gpu_ms,
        "accelerator_ms": accelerator_ms,
        "cpu_model_ms": 0.0,
        "cpu_host_ms": max(total_ms - accelerator_ms, 0.0),
        "cpu_non_model_ms": max(total_ms - accelerator_ms, 0.0),
        "total_runtime_ms": total_ms,
    }
    return result, warped


@torch.no_grad()
def measure_3d(
    namespace: dict,
    model,
    moving_raw: np.ndarray,
    fixed_raw: np.ndarray,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    total_started = time.perf_counter()
    moving_tensor = namespace["preprocess_volume_3d"](
        moving_raw, is_seg=False, device_override=device
    )
    fixed_tensor = namespace["preprocess_volume_3d"](
        fixed_raw, is_seg=False, device_override=device
    )

    def model_stage():
        model_output = model(moving_tensor, fixed_tensor)
        return (
            model_output[-1]
            if isinstance(model_output, (tuple, list))
            else model_output
        )

    flow_downsampled, model_ms = timed_cuda(model_stage, device)
    raw_flow = namespace["upsample_3d_flow_to_raw"](
        flow_downsampled[0].detach().cpu().numpy(), moving_raw.shape
    )
    warped, warp_gpu_ms = timed_cuda(
        lambda: namespace["warp_volume_3d"](
            moving_raw,
            raw_flow,
            mode="bilinear",
            runtime_device=device,
        ),
        device,
    )
    total_ms = (time.perf_counter() - total_started) * 1000.0
    accelerator_ms = model_ms + warp_gpu_ms
    result = {
        "model": "3d_gpu",
        "accelerator": "gpu",
        "model_inference_ms": model_ms,
        "accelerator_warp_ms": warp_gpu_ms,
        "accelerator_ms": accelerator_ms,
        "cpu_model_ms": 0.0,
        "cpu_host_ms": max(total_ms - accelerator_ms, 0.0),
        "cpu_non_model_ms": max(total_ms - accelerator_ms, 0.0),
        "total_runtime_ms": total_ms,
    }
    return result, warped


def aggregate(rows: list[dict], model: str) -> dict:
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


def main() -> None:
    os.environ["V4_MEASURE_LOCAL_POWER"] = "0"
    namespace = load_measurement_namespace()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")

    dataset = namespace["load_split"]("test", None)
    pairs = namespace["build_eval_pairs"](
        len(dataset), max_pairs=None, seed=namespace["PAIR_SEED"]
    )
    v4_model = namespace["load_board_cpu_reference"](
        namespace["V4_WEIGHTS"]
    ).to(device).eval()
    model_3d = namespace["VxmDense3DV2"]().to(device).eval()
    state = torch.load(
        namespace["V3D_WEIGHTS"], map_location="cpu", weights_only=False
    )
    model_3d.load_state_dict(state.get("state_dict", state))

    warm_moving, warm_fixed = load_raw_pair(namespace, dataset, *pairs[0])
    measure_v4_fused(namespace, v4_model, warm_moving, warm_fixed, device)
    measure_3d(namespace, model_3d, warm_moving, warm_fixed, device)

    rows = []
    for pair_index, (moving_idx, fixed_idx) in enumerate(pairs):
        moving_raw, fixed_raw = load_raw_pair(
            namespace, dataset, moving_idx, fixed_idx
        )
        for measure in (measure_v4_fused, measure_3d):
            model = v4_model if measure is measure_v4_fused else model_3d
            result, warped = measure(
                namespace, model, moving_raw, fixed_raw, device
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
            )
            rows.append(result)
        print(
            f"Pair {pair_index + 1}/{len(pairs)}:",
            {
                row["model"]: round(row["total_runtime_ms"], 2)
                for row in rows[-2:]
            },
        )

    models = {
        model: aggregate(rows, model)
        for model in ("2.5d_fused_gpu", "3d_gpu")
    }
    payload = {
        "schema_version": 1,
        "scope": {
            "start": (
                "normalized raw in-memory moving/fixed volumes enter "
                "model preparation"
            ),
            "end": "warped moving intensity volume is available in host memory",
            "included": (
                "model-input preparation, transfers, model execution, "
                "flow lifting/fusion or upsampling, and intensity-volume warping"
            ),
            "excluded": (
                "disk I/O, segmentation warping, Dice, TRE, MI, SSIM, "
                "power sampling, plotting, and report generation"
            ),
            "cpu_definition": (
                "total wall time minus synchronized CUDA model/warp device time"
            ),
        },
        "pair_count": len(pairs),
        "models": models,
        "pairs": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(models, indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
