#!/usr/bin/env python3
"""Create fixed-grid V4 calibration tensors from cross-subject volume pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VOXELMORPH_ROOT = REPO_ROOT / "Voxelmorph"
DEFAULT_DATA_ROOT = Path(os.environ.get("REGISTRATION_DATA_ROOT", str(REPO_ROOT / "Data" / "registration_dataset")))
for path in (REPO_ROOT, VOXELMORPH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_2p5d_pytorchV4 import load_split
from vxm_2p5d_v4 import CANONICAL_VOLUME_SHAPE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, N_STACK, ORIENTATIONS, combine_stacks, extract_stack, letterbox_stack


def normalize_to_contract(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    lo, hi = float(volume.min()), float(volume.max())
    if hi <= lo:
        return np.zeros_like(volume)
    if lo >= -1.001 and hi <= 1.001:
        return (2.0 * volume - 1.0) if lo >= -1e-4 else volume
    return 2.0 * (volume - lo) / (hi - lo) - 1.0


def load_volumes(data_root: Path):
    """Return lazy train fields; decoded volumes stay in a bounded cache."""
    moving, fixed, _, _ = load_split(data_root, "train", cache_subjects=16, cache_dir=data_root / ".v4_canonical_cache")
    return moving, fixed


def to_v4_stack(stack: np.ndarray) -> np.ndarray:
    if stack.shape[0] == N_STACK:
        return stack
    if stack.shape[0] != N_STACK - 1:
        raise ValueError(f"Expected {N_STACK - 1} or {N_STACK} stack slices, got {stack.shape[0]}")
    return np.concatenate([stack, stack[-1:, :, :]], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "calibration_data_v4_canonical")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    moving_volumes, fixed_volumes = load_volumes(args.data_root)
    if len(moving_volumes) < 2:
        raise ValueError("V4 calibration requires at least two training subjects")
    output_dir = args.output_dir / "inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []

    for index in range(args.num_samples):
        orientation = tuple(ORIENTATIONS)[int(rng.integers(len(ORIENTATIONS)))]
        axis = int(ORIENTATIONS[orientation]["axis"])
        moving_id = int(rng.integers(len(moving_volumes)))
        fixed_id = int(rng.integers(len(fixed_volumes)))
        while fixed_id == moving_id:
            fixed_id = int(rng.integers(len(fixed_volumes)))
        depth = min(moving_volumes[moving_id].shape[axis], fixed_volumes[fixed_id].shape[axis])
        z = int(rng.integers(3, depth - 3))
        moving_stack, moving_transform = letterbox_stack(to_v4_stack(extract_stack(moving_volumes[moving_id], orientation, z)), orientation)
        fixed_stack, fixed_transform = letterbox_stack(to_v4_stack(extract_stack(fixed_volumes[fixed_id], orientation, z)), orientation)
        np.save(output_dir / f"input_{index:04d}.npy", combine_stacks(moving_stack, fixed_stack))
        records.append({"orientation": orientation, "moving_id": moving_id, "fixed_id": fixed_id, "z": z, "moving_transform": moving_transform.to_dict(), "fixed_transform": fixed_transform.to_dict()})

    metadata = {
        "version": "v4-canonical-3d",
        "input_layout": "[channels,height,width] on disk; [batch,height,width,channels] for FPGA",
        "input_shape": [1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS],
        "canonical_volume_shape": list(CANONICAL_VOLUME_SHAPE),
        "n_stack": N_STACK,
        "pad_value": -1.0,
        "resize_policy": "trilinear image / nearest-label resample to canonical 3D grid, then orientation canonicalization and centered letterbox padding",
        "samples": records,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} fixed-grid V4 tensors to {output_dir}")


if __name__ == "__main__":
    main()
