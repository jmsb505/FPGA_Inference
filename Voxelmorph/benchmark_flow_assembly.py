"""Benchmark the CPU work required to assemble the fused 2.5D flow field.

This script intentionally excludes model execution, image warping, and quality
metrics. Run it once on the local host and once on the FPGA board so the chart
can report the same flow-assembly operation on x86 CPU and ARM CPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CANONICAL_SHAPE = (96, 112, 96)
WINDOW_RADIUS = 3


def lift_axial(flow_2d: np.ndarray) -> np.ndarray:
    field = np.zeros((3, *CANONICAL_SHAPE), dtype=np.float32)
    for index, z in enumerate(range(WINDOW_RADIUS, CANONICAL_SHAPE[0] - WINDOW_RADIUS)):
        field[0, z], field[1, z] = flow_2d[index, 0], flow_2d[index, 1]
    return field
def lift_coronal(flow_2d: np.ndarray) -> np.ndarray:
    field = np.zeros((3, *CANONICAL_SHAPE), dtype=np.float32)
    for index, y in enumerate(range(WINDOW_RADIUS, CANONICAL_SHAPE[1] - WINDOW_RADIUS)):
        field[0, :, y, :], field[2, :, y, :] = flow_2d[index, 0], flow_2d[index, 1]
    return field
def lift_sagittal(flow_2d: np.ndarray) -> np.ndarray:
    field = np.zeros((3, *CANONICAL_SHAPE), dtype=np.float32)
    for index, x in enumerate(range(WINDOW_RADIUS, CANONICAL_SHAPE[2] - WINDOW_RADIUS)):
        field[1, :, :, x], field[2, :, :, x] = flow_2d[index, 0], flow_2d[index, 1]
    return field
def assemble_mean_flow(
    axial: np.ndarray,
    coronal: np.ndarray,
    sagittal: np.ndarray,
) -> np.ndarray:
    axial_3d = lift_axial(axial)
    coronal_3d = lift_coronal(coronal)
    sagittal_3d = lift_sagittal(sagittal)
    fused = np.zeros_like(axial_3d)
    fused[0] = 0.5 * (axial_3d[0] + coronal_3d[0])
    fused[1] = 0.5 * (axial_3d[1] + sagittal_3d[1])
    fused[2] = 0.5 * (coronal_3d[2] + sagittal_3d[2])
    return fused
def benchmark(repeats: int, warmups: int) -> dict:
    rng = np.random.default_rng(42)
    axial = rng.standard_normal((90, 2, 112, 96), dtype=np.float32)
    coronal = rng.standard_normal((106, 2, 96, 96), dtype=np.float32)
    sagittal = rng.standard_normal((90, 2, 96, 112), dtype=np.float32)

    for _ in range(warmups):
        assemble_mean_flow(axial, coronal, sagittal)

    durations_ms = []
    for _ in range(repeats):
        started = time.perf_counter()
        fused = assemble_mean_flow(axial, coronal, sagittal)
        durations_ms.append((time.perf_counter() - started) * 1000.0)
        if fused.shape != (3, *CANONICAL_SHAPE):
            raise RuntimeError(f"Unexpected fused flow shape: {fused.shape}")

    values = np.asarray(durations_ms, dtype=np.float64)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "canonical_shape": list(CANONICAL_SHAPE),
        "operation": "three orientation lifts plus arithmetic mean fusion",
        "scope": "CPU flow assembly only; excludes inference, warping, and metrics",
        "warmups": warmups,
        "repeats": repeats,
        "cpu_flow_assembly_ms_mean": float(values.mean()),
        "cpu_flow_assembly_ms_sd": float(values.std(ddof=1)),
        "cpu_flow_assembly_ms_median": float(np.median(values)),
        "durations_ms": durations_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = benchmark(args.repeats, args.warmups)
    result["label"] = args.label
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{args.label}: {result['cpu_flow_assembly_ms_mean']:.3f} "
        f"+/- {result['cpu_flow_assembly_ms_sd']:.3f} ms"
    )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
