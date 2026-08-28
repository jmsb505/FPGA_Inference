"""V4 2.5D VoxelMorph core and fixed-grid preprocessing contract.

V4 has one FPGA-facing input: ``[B, 112, 96, 16]``.  The 16 channels are
eight moving slices followed by eight fixed slices.  Every training,
calibration, local-inference, and FPGA caller must use the helpers in this
module so the model never sees the V2 orientation-dependent input sizes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_HEIGHT = 112
INPUT_WIDTH = 96
N_STACK = 8
INPUT_CHANNELS = N_STACK * 2
PAD_VALUE = -1.0
FLOW_SCALE = 0.1

# Volumes are standardized before plane extraction. This makes the three
# orientations describe a consistent anatomical field of view: axial is
# 112x96, coronal is 96x96, and sagittal becomes 112x96 after transposition.
CANONICAL_VOLUME_SHAPE = (96, 112, 96)  # [depth, height, width]
ORIENTATIONS: Dict[str, Dict[str, object]] = {
    "axial": {"axis": 0, "transpose_hw": False},
    "coronal": {"axis": 1, "transpose_hw": False},
    # A sagittal stack is [96, 112].  Transposing it gives the V4 frame
    # without interpolation and keeps the common model contract explicit.
    "sagittal": {"axis": 2, "transpose_hw": True},
}


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometry needed to map V4 predictions back to an input plane."""

    orientation: str
    source_hw: Tuple[int, int]
    canonical_hw: Tuple[int, int]
    resized_hw: Tuple[int, int]
    scale: float
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    transposed: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def valid_mask(self) -> np.ndarray:
        mask = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
        h, w = self.resized_hw
        mask[self.pad_top:self.pad_top + h, self.pad_left:self.pad_left + w] = 1.0
        return mask


def orientation_spec(orientation: str) -> Dict[str, object]:
    try:
        return ORIENTATIONS[orientation]
    except KeyError as exc:
        raise ValueError(f"Unknown orientation {orientation!r}; expected one of {tuple(ORIENTATIONS)}") from exc



def resample_volume(
    volume: np.ndarray,
    *,
    is_segmentation: bool = False,
    target_shape: Tuple[int, int, int] = CANONICAL_VOLUME_SHAPE,
) -> np.ndarray:
    """Resample a [D,H,W] volume to V4's shared canonical 3D grid.

    Images use trilinear interpolation; discrete segmentations use nearest
    neighbor interpolation so label IDs are never blended.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected volume [D,H,W], got {volume.shape}")
    target_shape = tuple(int(dimension) for dimension in target_shape)
    if len(target_shape) != 3 or any(dimension <= 0 for dimension in target_shape):
        raise ValueError(f"Invalid canonical volume shape: {target_shape}")
    if tuple(volume.shape) == target_shape:
        return np.ascontiguousarray(volume.astype(np.int64 if is_segmentation else np.float32, copy=False))

    tensor = torch.from_numpy(np.ascontiguousarray(volume.astype(np.float32, copy=False))).unsqueeze(0).unsqueeze(0)
    if is_segmentation:
        resized = F.interpolate(tensor, size=target_shape, mode="nearest")
        return np.ascontiguousarray(resized[0, 0].numpy().astype(np.int64))
    resized = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    return np.ascontiguousarray(resized[0, 0].numpy().astype(np.float32))

def extract_stack(volume: np.ndarray, orientation: str, z: int, window_radius: int = 3) -> np.ndarray:
    """Return a native seven-slice stack in the orientation's plane."""
    axis = int(orientation_spec(orientation)["axis"])
    if axis == 0:
        out = volume[z - window_radius:z + window_radius + 1]
    elif axis == 1:
        out = volume[:, z - window_radius:z + window_radius + 1, :].transpose(1, 0, 2)
    else:
        out = volume[:, :, z - window_radius:z + window_radius + 1].transpose(2, 0, 1)
    return np.ascontiguousarray(out)


def extract_plane(volume: np.ndarray, orientation: str, z: int) -> np.ndarray:
    axis = int(orientation_spec(orientation)["axis"])
    if axis == 0:
        return np.ascontiguousarray(volume[z])
    if axis == 1:
        return np.ascontiguousarray(volume[:, z, :])
    return np.ascontiguousarray(volume[:, :, z])


def _canonicalize(array: np.ndarray, orientation: str) -> np.ndarray:
    """Put a stack ``[N,H,W]`` or plane ``[H,W]`` into the V4 axis order."""
    if bool(orientation_spec(orientation)["transpose_hw"]):
        return np.ascontiguousarray(np.swapaxes(array, -2, -1))
    return np.ascontiguousarray(array)


def _resize_last_two(array: np.ndarray, target_hw: Tuple[int, int], interpolation: int) -> np.ndarray:
    target_h, target_w = target_hw
    if array.shape[-2:] == target_hw:
        return np.ascontiguousarray(array)
    if array.ndim == 2:
        return cv2.resize(array, (target_w, target_h), interpolation=interpolation)
    if array.ndim != 3:
        raise ValueError(f"Expected [H,W] or [N,H,W], got {array.shape}")
    return np.ascontiguousarray(np.stack([
        cv2.resize(array[index], (target_w, target_h), interpolation=interpolation)
        for index in range(array.shape[0])
    ]))


def make_letterbox_transform(source_hw: Tuple[int, int], orientation: str) -> LetterboxTransform:
    """Create an aspect-ratio-preserving transform into the V4 grid."""
    canonical_hw = source_hw[::-1] if bool(orientation_spec(orientation)["transpose_hw"]) else source_hw
    source_h, source_w = canonical_hw
    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"Invalid input plane size: {source_hw}")
    scale = min(INPUT_HEIGHT / source_h, INPUT_WIDTH / source_w)
    resized_h = min(INPUT_HEIGHT, max(1, int(round(source_h * scale))))
    resized_w = min(INPUT_WIDTH, max(1, int(round(source_w * scale))))
    pad_h = INPUT_HEIGHT - resized_h
    pad_w = INPUT_WIDTH - resized_w
    return LetterboxTransform(
        orientation=orientation,
        source_hw=tuple(int(x) for x in source_hw),
        canonical_hw=tuple(int(x) for x in canonical_hw),
        resized_hw=(resized_h, resized_w),
        scale=float(scale),
        pad_top=pad_h // 2,
        pad_bottom=pad_h - pad_h // 2,
        pad_left=pad_w // 2,
        pad_right=pad_w - pad_w // 2,
        transposed=bool(orientation_spec(orientation)["transpose_hw"]),
    )


def apply_letterbox(
    array: np.ndarray,
    transform: LetterboxTransform,
    *,
    pad_value: float,
    interpolation: int,
) -> np.ndarray:
    """Apply an existing transform to images, stacks, or label planes."""
    canonical = _canonicalize(array, transform.orientation)
    if tuple(canonical.shape[-2:]) != transform.canonical_hw:
        raise ValueError(
            f"Input shape {array.shape} does not match {transform.orientation} transform "
            f"for source plane {transform.source_hw}"
        )
    resized = _resize_last_two(canonical, transform.resized_hw, interpolation)
    out_shape = (*resized.shape[:-2], INPUT_HEIGHT, INPUT_WIDTH)
    out = np.full(out_shape, pad_value, dtype=resized.dtype)
    h, w = transform.resized_hw
    out[..., transform.pad_top:transform.pad_top + h, transform.pad_left:transform.pad_left + w] = resized
    return np.ascontiguousarray(out)


def letterbox_stack(stack: np.ndarray, orientation: str, pad_value: float = PAD_VALUE) -> Tuple[np.ndarray, LetterboxTransform]:
    """Letterbox an image stack into ``[N,112,96]`` using linear interpolation."""
    if stack.ndim != 3:
        raise ValueError(f"Expected stack [N,H,W], got {stack.shape}")
    # The historical training window has seven slices. V4 makes the eighth
    # edge slice part of its declared input contract before any training or
    # calibration transform is applied.
    if stack.shape[0] == N_STACK - 1:
        stack = np.concatenate([stack, stack[-1:, :, :]], axis=0)
    elif stack.shape[0] != N_STACK:
        raise ValueError(f"Expected {N_STACK - 1} or {N_STACK} stack slices, got {stack.shape[0]}")
    transform = make_letterbox_transform(tuple(stack.shape[-2:]), orientation)
    result = apply_letterbox(stack.astype(np.float32, copy=False), transform, pad_value=pad_value, interpolation=cv2.INTER_LINEAR)
    return result.astype(np.float32, copy=False), transform


def letterbox_plane(
    plane: np.ndarray,
    transform: LetterboxTransform,
    *,
    is_segmentation: bool = False,
) -> np.ndarray:
    """Map a matching centre image or segmentation to the stack's V4 grid."""
    interpolation = cv2.INTER_NEAREST if is_segmentation else cv2.INTER_LINEAR
    pad_value = 0 if is_segmentation else PAD_VALUE
    result = apply_letterbox(plane, transform, pad_value=pad_value, interpolation=interpolation)
    return result.astype(np.int64 if is_segmentation else np.float32, copy=False)


def combine_stacks(moving_stack: np.ndarray, fixed_stack: np.ndarray) -> np.ndarray:
    """Return a channel-first V4 sample ``[16,112,96]``."""
    if moving_stack.shape != (N_STACK, INPUT_HEIGHT, INPUT_WIDTH):
        raise ValueError(f"Invalid moving stack shape: {moving_stack.shape}")
    if fixed_stack.shape != moving_stack.shape:
        raise ValueError(f"Moving/fixed stack mismatch: {moving_stack.shape} vs {fixed_stack.shape}")
    return np.ascontiguousarray(np.concatenate([moving_stack, fixed_stack], axis=0).astype(np.float32))


def prepare_pair(
    moving_stack: np.ndarray,
    fixed_stack: np.ndarray,
    moving_center: np.ndarray,
    fixed_center: np.ndarray,
    moving_seg: np.ndarray,
    fixed_seg: np.ndarray,
    orientation: str,
) -> Dict[str, object]:
    """Prepare one complete V4 training sample and its invertible metadata."""
    moving_stack_v4, moving_transform = letterbox_stack(moving_stack, orientation)
    fixed_stack_v4, fixed_transform = letterbox_stack(fixed_stack, orientation)
    moving_mask = moving_transform.valid_mask()
    fixed_mask = fixed_transform.valid_mask()
    return {
        "input": combine_stacks(moving_stack_v4, fixed_stack_v4),
        "moving_center": letterbox_plane(moving_center, moving_transform),
        "fixed_center": letterbox_plane(fixed_center, fixed_transform),
        "moving_seg": letterbox_plane(moving_seg, moving_transform, is_segmentation=True),
        "fixed_seg": letterbox_plane(fixed_seg, fixed_transform, is_segmentation=True),
        "valid_mask": np.minimum(moving_mask, fixed_mask).astype(np.float32),
        "moving_transform": moving_transform.to_dict(),
        "fixed_transform": fixed_transform.to_dict(),
    }


class Vxm2p5dV4(nn.Module):
    """One-model V4 core with a strict channel-last FPGA input contract."""

    def __init__(self, enc_feats: Tuple[int, ...] = (16, 32, 32, 32), final_feats: Tuple[int, int] = (32, 16)):
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        in_channels = INPUT_CHANNELS
        for features in enc_feats:
            self.enc_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(0.1),
            ))
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(features, features, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(0.1),
            ))
            in_channels = features

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        in_channels = 32
        for skip_channels in reversed(enc_feats):
            self.up_blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels + skip_channels, skip_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(skip_channels),
                nn.LeakyReLU(0.1),
            ))
            in_channels = skip_channels

        self.final_conv0 = nn.Sequential(nn.Conv2d(in_channels, final_feats[0], 3, padding=1, bias=False), nn.BatchNorm2d(final_feats[0]), nn.LeakyReLU(0.1))
        self.final_conv1 = nn.Sequential(nn.Conv2d(final_feats[0], final_feats[1], 3, padding=1, bias=False), nn.BatchNorm2d(final_feats[1]), nn.LeakyReLU(0.1))
        self.flow_unscaled = nn.Conv2d(final_feats[1], 2, 3, padding=1)
        nn.init.zeros_(self.flow_unscaled.weight)
        nn.init.zeros_(self.flow_unscaled.bias)
        self.flow_scale = nn.Conv2d(2, 2, 1, bias=False)
        self.flow_scale.weight.data.zero_()
        self.flow_scale.weight.data[0, 0, 0, 0] = FLOW_SCALE
        self.flow_scale.weight.data[1, 1, 0, 0] = FLOW_SCALE
        self.flow_scale.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS):
            raise ValueError(f"V4 expects [B,{INPUT_HEIGHT},{INPUT_WIDTH},{INPUT_CHANNELS}], got {tuple(x.shape)}")
        x = x.permute(0, 3, 1, 2)
        skips = []
        for encoder, downsample in zip(self.enc_blocks, self.down_blocks):
            x = encoder(x)
            skips.append(x)
            x = downsample(x)
        x = self.bottleneck(x)
        for upsample, decoder, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            x = upsample(x)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.flow_scale(self.flow_unscaled(self.final_conv1(self.final_conv0(x))))


def load_model_for_export(checkpoint: Path, config=None, class_name: str = "Vxm2p5dV4") -> Vxm2p5dV4:
    if class_name != "Vxm2p5dV4":
        raise ValueError(f"Unsupported V4 export class: {class_name}")
    state = torch.load(Path(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {checkpoint}")
    model = Vxm2p5dV4()
    model.load_state_dict(state, strict=True)
    return model.eval()


def get_calib_loader(config):
    """Load already-letterboxed V4 calibration tensors from ``input_*.npy``."""
    expected_shape = (1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
    if getattr(config, "input_shape", None) and tuple(int(x) for x in str(config.input_shape).split(",")) != expected_shape:
        raise ValueError(f"V4 requires input_shape {expected_shape}, got {config.input_shape}")
    inputs_dir = Path(config.calib_dir) / "inputs"
    files = sorted(inputs_dir.glob("input_*.npy"))
    if not files:
        raise FileNotFoundError(f"No V4 calibration tensors found in {inputs_dir}")
    if getattr(config, "calib_samples", None):
        files = files[:int(config.calib_samples)]
    loader = []
    for path in files:
        array = np.load(path).astype(np.float32)
        if array.shape != (INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH):
            raise ValueError(f"{path} has shape {array.shape}; expected {(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)}")
        loader.append(torch.from_numpy(array).unsqueeze(0).permute(0, 2, 3, 1))
    return loader
