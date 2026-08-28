#!/usr/bin/env python3
"""Train the fixed-grid, letterboxed V4 2.5D VoxelMorph model.

Unlike V2, every orientation is transformed to the V4 FPGA input contract
before it reaches the network: [batch, 112, 96, 16] (NHWC).
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
VOXELMORPH_ROOT = REPO_ROOT / "Voxelmorph"
for path in (REPO_ROOT, VOXELMORPH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from registration_dataset import RegistrationDataset
from vxm_2p5d_v4 import CANONICAL_VOLUME_SHAPE, INPUT_HEIGHT, INPUT_WIDTH, N_STACK, ORIENTATIONS, Vxm2p5dV4, combine_stacks, extract_plane, extract_stack, letterbox_plane, letterbox_stack, resample_volume


SEG_LABELS = (2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28)
WINDOW_RADIUS = 3
DEFAULT_DATA_ROOT = Path(os.environ.get("REGISTRATION_DATA_ROOT", str(REPO_ROOT / "Data" / "registration_dataset")))


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    lo, hi = float(volume.min()), float(volume.max())
    if hi <= lo:
        return np.zeros_like(volume)
    if lo >= -1.001 and hi <= 1.001:
        return (2.0 * volume - 1.0) if lo >= -1e-4 else volume
    return 2.0 * (volume - lo) / (hi - lo) - 1.0


CACHE_FORMAT_VERSION = "v4-canonical-3d-cache-v1"
Subject = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class _LazyDatasetSplit:
    """Load V4 subjects from a persistent canonical cache with a bounded RAM LRU."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        max_subjects: int | None,
        cache_subjects: int,
        cache_dir: Path | None = None,
    ):
        self.dataset = RegistrationDataset(str(data_root), split=split)
        count = len(self.dataset) if max_subjects is None else min(max_subjects, len(self.dataset))
        self.indices = tuple(range(count))
        self.cache_subjects = max(0, int(cache_subjects))
        self.cache_dir = (Path(cache_dir) / split) if cache_dir is not None else None
        self._cache: OrderedDict[int, Subject] = OrderedDict()

    def __len__(self) -> int:
        return len(self.indices)

    def _cache_path(self, local_index: int) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"subject_{self.indices[local_index]:04d}.npz"

    @staticmethod
    def _validate_subject(subject: Subject) -> Subject | None:
        images = subject[:2]
        labels = subject[2:]
        if any(array.shape != CANONICAL_VOLUME_SHAPE for array in subject):
            return None
        if any(array.dtype != np.float32 for array in images):
            return None
        if any(not np.issubdtype(array.dtype, np.integer) for array in labels):
            return None
        return (
            np.ascontiguousarray(images[0]),
            np.ascontiguousarray(images[1]),
            np.ascontiguousarray(labels[0].astype(np.int64, copy=False)),
            np.ascontiguousarray(labels[1].astype(np.int64, copy=False)),
        )

    def _read_disk_cache(self, local_index: int) -> Subject | None:
        cache_path = self._cache_path(local_index)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["version"].item()) != CACHE_FORMAT_VERSION:
                    return None
                subject = (
                    cached["moving"].astype(np.float32, copy=False),
                    cached["fixed"].astype(np.float32, copy=False),
                    cached["moving_seg"],
                    cached["fixed_seg"],
                )
            return self._validate_subject(subject)
        except (KeyError, OSError, ValueError):
            return None

    def _write_disk_cache(self, local_index: int, subject: Subject) -> None:
        cache_path = self._cache_path(local_index)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(f"{cache_path.stem}.tmp.npz")
        np.savez_compressed(
            temporary_path,
            version=np.array(CACHE_FORMAT_VERSION),
            moving=subject[0],
            fixed=subject[1],
            moving_seg=subject[2].astype(np.int16, copy=False),
            fixed_seg=subject[3].astype(np.int16, copy=False),
        )
        os.replace(temporary_path, cache_path)

    def _build_subject(self, local_index: int) -> Subject:
        sample = self.dataset[self.indices[local_index]]
        if "moving_seg" not in sample or "fixed_seg" not in sample:
            raise ValueError("V4 supervised training requires moving and fixed segmentations")
        subject = (
            normalize_volume(resample_volume(sample["moving"].squeeze(0).numpy())),
            normalize_volume(resample_volume(sample["fixed"].squeeze(0).numpy())),
            resample_volume(sample["moving_seg"].squeeze(0).numpy(), is_segmentation=True),
            resample_volume(sample["fixed_seg"].squeeze(0).numpy(), is_segmentation=True),
        )
        validated = self._validate_subject(subject)
        if validated is None:
            raise RuntimeError("V4 canonical preprocessing produced an invalid subject cache entry")
        return validated

    def load_subject(self, local_index: int) -> Subject:
        if local_index < 0 or local_index >= len(self):
            raise IndexError(local_index)
        if local_index in self._cache:
            self._cache.move_to_end(local_index)
            return self._cache[local_index]

        subject = self._read_disk_cache(local_index)
        if subject is None:
            subject = self._build_subject(local_index)
            self._write_disk_cache(local_index, subject)
        if self.cache_subjects:
            self._cache[local_index] = subject
            self._cache.move_to_end(local_index)
            while len(self._cache) > self.cache_subjects:
                self._cache.popitem(last=False)
        return subject


class _LazyField:
    def __init__(self, split: _LazyDatasetSplit, field_index: int):
        self.split = split
        self.field_index = field_index

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.split.load_subject(int(index))[self.field_index]


def load_split(
    data_root: Path,
    split: str,
    max_subjects: int | None = None,
    cache_subjects: int = 16,
    cache_dir: Path | None = None,
) -> Tuple[_LazyField, _LazyField, _LazyField, _LazyField]:
    """Return lazy fields backed by canonical on-disk and bounded RAM caches."""
    lazy_split = _LazyDatasetSplit(data_root, split, max_subjects, cache_subjects, cache_dir)
    if len(lazy_split) < 2:
        raise ValueError(f"V4 training requires at least two subjects in the {split!r} split")
    return tuple(_LazyField(lazy_split, field_index) for field_index in range(4))


def build_canonical_cache(
    data_root: Path,
    split: str,
    cache_dir: Path,
    max_subjects: int | None = None,
) -> None:
    """Materialize V4-normalized canonical volumes once for fast later epochs."""
    lazy_split = _LazyDatasetSplit(data_root, split, max_subjects, cache_subjects=0, cache_dir=cache_dir)
    print(f"Preparing {len(lazy_split)} canonical V4 {split} subjects in {lazy_split.cache_dir}")
    for local_index in range(len(lazy_split)):
        lazy_split.load_subject(local_index)
        if (local_index + 1) % 10 == 0 or local_index + 1 == len(lazy_split):
            print(f"  cached {local_index + 1}/{len(lazy_split)}")


def to_v4_stack(stack: np.ndarray) -> np.ndarray:
    """Make V4's eight-slice window explicit during training and calibration."""
    if stack.shape[0] == N_STACK:
        return stack
    if stack.shape[0] != N_STACK - 1:
        raise ValueError(f"Expected {N_STACK - 1} or {N_STACK} stack slices, got {stack.shape[0]}")
    return np.concatenate([stack, stack[-1:, :, :]], axis=0)


def sample_subject_ids(count: int, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    moving_ids = rng.integers(count, size=batch_size)
    fixed_ids = rng.integers(count, size=batch_size)
    for index in range(batch_size):
        while fixed_ids[index] == moving_ids[index]:
            fixed_ids[index] = rng.integers(count)
    return moving_ids, fixed_ids


def build_batch(volumes, segmentations, batch_size: int, rng: np.random.Generator) -> Dict[str, object]:
    moving_volumes, fixed_volumes = volumes
    moving_segs, fixed_segs = segmentations
    orientation = tuple(ORIENTATIONS)[int(rng.integers(len(ORIENTATIONS)))]
    axis = int(ORIENTATIONS[orientation]["axis"])
    moving_ids, fixed_ids = sample_subject_ids(len(moving_volumes), batch_size, rng)
    inputs: List[np.ndarray] = []
    moving_centers: List[np.ndarray] = []
    fixed_centers: List[np.ndarray] = []
    moving_labels: List[np.ndarray] = []
    fixed_labels: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for moving_id, fixed_id in zip(moving_ids, fixed_ids):
        depth = min(moving_volumes[moving_id].shape[axis], fixed_volumes[fixed_id].shape[axis])
        if depth <= 2 * WINDOW_RADIUS:
            raise ValueError(f"Volume is too shallow for {orientation}: {depth}")
        z = int(rng.integers(WINDOW_RADIUS, depth - WINDOW_RADIUS))
        moving_stack, moving_transform = letterbox_stack(to_v4_stack(extract_stack(moving_volumes[moving_id], orientation, z)), orientation)
        fixed_stack, fixed_transform = letterbox_stack(to_v4_stack(extract_stack(fixed_volumes[fixed_id], orientation, z)), orientation)
        inputs.append(combine_stacks(moving_stack, fixed_stack))
        moving_centers.append(letterbox_plane(extract_plane(moving_volumes[moving_id], orientation, z), moving_transform))
        fixed_centers.append(letterbox_plane(extract_plane(fixed_volumes[fixed_id], orientation, z), fixed_transform))
        moving_labels.append(letterbox_plane(extract_plane(moving_segs[moving_id], orientation, z), moving_transform, is_segmentation=True))
        fixed_labels.append(letterbox_plane(extract_plane(fixed_segs[fixed_id], orientation, z), fixed_transform, is_segmentation=True))
        masks.append(np.minimum(moving_transform.valid_mask(), fixed_transform.valid_mask()))

    return {
        "input": np.stack(inputs).astype(np.float32),
        "moving_center": np.stack(moving_centers).astype(np.float32),
        "fixed_center": np.stack(fixed_centers).astype(np.float32),
        "moving_seg": np.stack(moving_labels).astype(np.int64),
        "fixed_seg": np.stack(fixed_labels).astype(np.int64),
        "valid_mask": np.stack(masks).astype(np.float32),
        "orientation": orientation,
    }


def spatial_transform(source: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = source.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=source.device), torch.arange(width, device=source.device), indexing="ij")
    grid = torch.stack((xx, yy), dim=0).float().unsqueeze(0).expand(batch, -1, -1, -1) + flow
    grid[:, 0] = 2.0 * grid[:, 0] / (width - 1.0) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / (height - 1.0) - 1.0
    return F.grid_sample(source, grid.permute(0, 2, 3, 1), align_corners=True, mode="bilinear", padding_mode="border")


def one_hot(segmentation: torch.Tensor) -> torch.Tensor:
    return torch.stack([(segmentation == label) for label in SEG_LABELS], dim=1).float()


def dice_loss(warped: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(1)
    inter = (warped * fixed * mask).sum(dim=(0, 2, 3))
    size = (warped * mask).sum(dim=(0, 2, 3)) + (fixed * mask).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * inter + 1e-5) / (size + 1e-5)).mean()


def smoothness_loss(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x_mask = mask[:, :, 1:] * mask[:, :, :-1]
    y_mask = mask[:, 1:, :] * mask[:, :-1, :]
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).square() * x_mask.unsqueeze(1)
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).square() * y_mask.unsqueeze(1)
    return dx.sum() / (x_mask.sum() * flow.shape[1] + 1e-6) + dy.sum() / (y_mask.sum() * flow.shape[1] + 1e-6)


def masked_mi_loss(moving: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor, bins: int = 32) -> torch.Tensor:
    """Differentiable histogram MI, excluding letterbox padding."""
    centers = torch.linspace(-1.0, 1.0, bins, device=moving.device)
    sigma = 2.0 / (bins - 1) * 0.5
    moving_codes = torch.exp(-0.5 * ((moving.flatten(1).unsqueeze(-1) - centers) / sigma).square())
    fixed_codes = torch.exp(-0.5 * ((fixed.flatten(1).unsqueeze(-1) - centers) / sigma).square())
    moving_codes = moving_codes / (moving_codes.sum(dim=-1, keepdim=True) + 1e-6)
    fixed_codes = fixed_codes / (fixed_codes.sum(dim=-1, keepdim=True) + 1e-6)
    weights = mask.flatten(1).unsqueeze(-1)
    normalizer = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    joint = torch.bmm((moving_codes * weights).transpose(1, 2), fixed_codes) / normalizer
    moving_marginal = (moving_codes * weights).sum(dim=1, keepdim=True) / normalizer
    fixed_marginal = (fixed_codes * weights).sum(dim=1, keepdim=True) / normalizer
    independent = torch.bmm(moving_marginal.transpose(1, 2), fixed_marginal).clamp_min(1e-6)
    return -(joint * torch.log(joint.clamp_min(1e-6) / independent)).sum(dim=(1, 2)).mean()


def tensor_batch(batch: Dict[str, object], device: torch.device):
    model_input = torch.from_numpy(batch["input"]).permute(0, 2, 3, 1).to(device)
    return (
        model_input,
        torch.from_numpy(batch["moving_center"]).unsqueeze(1).to(device),
        torch.from_numpy(batch["fixed_center"]).unsqueeze(1).to(device),
        torch.from_numpy(batch["moving_seg"]).to(device),
        torch.from_numpy(batch["fixed_seg"]).to(device),
        torch.from_numpy(batch["valid_mask"]).to(device),
    )


def run_step(model, optimizer, batch, device, mi_weight: float, smooth_weight: float, train: bool) -> Dict[str, float]:
    model_input, moving, fixed, moving_seg, fixed_seg, mask = tensor_batch(batch, device)
    with torch.set_grad_enabled(train):
        flow = model(model_input)
        warped_image = spatial_transform(moving, flow)
        warped_segmentation = spatial_transform(one_hot(moving_seg), flow)
        dice = dice_loss(warped_segmentation, one_hot(fixed_seg), mask)
        mi = masked_mi_loss(warped_image[:, 0], fixed[:, 0], mask)
        smooth = smoothness_loss(flow, mask)
        total = dice + mi_weight * mi + smooth_weight * smooth
        if train and torch.isfinite(total):
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
    return {"total": float(total.detach()), "dice": float(dice.detach()), "mi": float(mi.detach()), "smooth": float(smooth.detach())}


def average(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--val-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mi-weight", type=float, default=0.5)
    parser.add_argument("--smooth-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--cache-subjects", type=int, default=16, help="Canonical subject pairs retained in RAM per split")
    parser.add_argument("--cache-dir", type=Path, help="Persistent canonical V4 cache directory")
    parser.add_argument("--no-disk-cache", action="store_true", help="Disable the persistent canonical cache")
    parser.add_argument("--build-cache", action="store_true", help="Materialize canonical cache before training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    weights_dir = VOXELMORPH_ROOT / "trained_weights"
    checkpoints_dir = weights_dir / "2p5d_pt_v4_canonical_checkpoints"
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "2p5d_dense_pt_v4_canonical_best.pth"
    latest_path = checkpoints_dir / "2p5d_pt_v4_canonical_latest.pth"

    cache_dir = None if args.no_disk_cache else (args.cache_dir or args.data_root / ".v4_canonical_cache")
    if args.build_cache and cache_dir is not None:
        build_canonical_cache(args.data_root, "train", cache_dir, args.max_train)
        build_canonical_cache(args.data_root, "val", cache_dir, args.max_val)
    train_volumes = load_split(args.data_root, "train", args.max_train, args.cache_subjects, cache_dir)
    val_volumes = load_split(args.data_root, "val", args.max_val, args.cache_subjects, cache_dir)
    train_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 11)
    val_batches = [build_batch(val_volumes[:2], val_volumes[2:], args.batch_size, val_rng) for _ in range(args.val_steps)]
    model = Vxm2p5dV4().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch, best_loss = 0, float("inf")
    if latest_path.exists() and not args.no_resume:
        state = torch.load(latest_path, map_location=device)
        if state.get("version") == "v4-canonical-3d":
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch, best_loss = int(state["epoch"]), float(state["best_loss"])
            print(f"Resumed V4 from epoch {start_epoch}")

    spec = {"version": "v4-canonical-3d", "canonical_volume_shape": list(CANONICAL_VOLUME_SHAPE), "input_shape": [1, INPUT_HEIGHT, INPUT_WIDTH, N_STACK * 2], "orientations": ORIENTATIONS, "pad_value": -1.0, "resize_policy": "trilinear image / nearest-label resample to canonical 3D grid, then centered letterbox padding"}
    (checkpoints_dir / "v4_model_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_rows = [run_step(model, optimizer, build_batch(train_volumes[:2], train_volumes[2:], args.batch_size, train_rng), device, args.mi_weight, args.smooth_weight, True) for _ in range(args.steps_per_epoch)]
        model.eval()
        with torch.no_grad():
            val_rows = [run_step(model, optimizer, batch, device, args.mi_weight, args.smooth_weight, False) for batch in val_batches]
        train_metrics, val_metrics = average(train_rows), average(val_rows)
        print(f"Epoch {epoch + 1}/{args.epochs} train={train_metrics['total']:.5f} val={val_metrics['total']:.5f}")
        if val_metrics["total"] < best_loss:
            best_loss = val_metrics["total"]
            torch.save(model.state_dict(), best_path)
        torch.save({"version": "v4-canonical-3d", "epoch": epoch + 1, "best_loss": best_loss, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "train_metrics": train_metrics, "val_metrics": val_metrics, "spec": spec}, latest_path)


if __name__ == "__main__":
    main()
