from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


EXPORT_FLOW_BOOST = 2.0


class Vxm2p5dDenseCore(nn.Module):
    def __init__(
        self,
        n_stack=8,
        enc_feats=(16, 32, 32, 32),
        final_feats=(32, 16),
        flow_scale=0.1,
        export_flow_boost=EXPORT_FLOW_BOOST,
    ):
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        in_ch = n_stack * 2
        self.export_flow_boost = float(export_flow_boost)

        for nf in enc_feats:
            self.enc_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, nf, 3, padding=1, bias=False),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.1),
            ))
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(nf, nf, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.1),
            ))
            in_ch = nf

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )

        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        in_ch = 32
        for skip_ch in reversed(enc_feats):
            self.up_blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, skip_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(skip_ch),
                nn.LeakyReLU(0.1),
            ))
            in_ch = skip_ch

        self.final_conv0 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        self.final_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
        )

        self.flow_unscaled = nn.Conv2d(16, 2, 3, padding=1)
        nn.init.zeros_(self.flow_unscaled.weight)
        nn.init.zeros_(self.flow_unscaled.bias)

        self.flow_scale = nn.Conv2d(2, 2, 1, bias=False)
        self.flow_scale.weight.data.zero_()
        self.flow_scale.weight.data[0, 0, 0, 0] = flow_scale
        self.flow_scale.weight.data[1, 1, 0, 0] = flow_scale
        self.flow_scale.weight.requires_grad = False

    def forward(self, x):
        if x.shape[1] != self.enc_blocks[0][0].in_channels:
            x = x.permute(0, 3, 1, 2)

        skips = []
        for enc, ds in zip(self.enc_blocks, self.down_blocks):
            x = enc(x)
            skips.append(x)
            x = ds(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.final_conv0(x)
        x = self.final_conv1(x)

        flow = self.flow_unscaled(x)
        if self.export_flow_boost != 1.0:
            flow = flow * self.export_flow_boost
        return self.flow_scale(flow)


def _unwrap_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def _pad_first_layer_if_needed(state, model):
    key = "enc_blocks.0.0.weight"
    if key not in state:
        return state
    ckpt_w = state[key]
    model_w = model.enc_blocks[0][0].weight
    if ckpt_w.shape[1] != 14 or model_w.shape[1] != 16:
        return state
    new_w = torch.zeros_like(model_w)
    new_w[:, :14, :, :] = ckpt_w
    state = dict(state)
    state[key] = new_w
    return state


def load_model_for_export(checkpoint, config=None, class_name="Vxm2p5dDenseCore"):
    if class_name != "Vxm2p5dDenseCore":
        raise ValueError(f"Unsupported 2.5D export class: {class_name}")

    model = Vxm2p5dDenseCore()
    state = torch.load(Path(checkpoint), map_location="cpu")
    state = _unwrap_state_dict(state)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {checkpoint}")

    state = _pad_first_layer_if_needed(state, model)
    model.load_state_dict(state, strict=True)

    if model.export_flow_boost != 1.0:
        with torch.no_grad():
            model.flow_scale.weight.div_(model.export_flow_boost)

    model.eval()
    return model


def _normalize_stack_to_contract(stack, norm_counts):
    arr = stack.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())

    if arr_max <= arr_min:
        norm_counts["constant_zero"] += 1
        return np.zeros_like(arr, dtype=np.float32)

    if arr_min >= -1.001 and arr_max <= 1.001:
        if arr_min >= -1e-4 and arr_max <= 1.0001:
            norm_counts["zero_one_to_minus_one_one"] += 1
            return (2.0 * arr - 1.0).astype(np.float32)
        norm_counts["already_-1_1"] += 1
        return arr

    norm_counts["minmax_to_minus_one_one"] += 1
    return (2.0 * (arr - arr_min) / (arr_max - arr_min) - 1.0).astype(np.float32)


def _pad_stack(stack):
    if stack.shape[0] == 7:
        return np.concatenate([stack, stack[-1:, :, :]], axis=0)
    return stack


def _pair_score(mv_arr, fx_arr):
    mv_dyn = float(np.max(mv_arr) - np.min(mv_arr))
    fx_dyn = float(np.max(fx_arr) - np.min(fx_arr))
    dyn = mv_dyn + fx_dyn
    var = float(np.var(mv_arr) + np.var(fx_arr))
    return dyn, var, dyn * (var + 1e-8)


def _select_indices(stats, requested):
    min_dynamic = 0.2
    min_variance = 1e-4
    valid = [i for i, stat in enumerate(stats) if stat["dyn"] >= min_dynamic and stat["var"] >= min_variance]
    if not valid:
        valid = list(range(len(stats)))
        print("[VXM-2P5D-EXPORT] Warning: no calibration samples met dynamic/variance threshold; using all samples")

    valid = sorted(valid, key=lambda i: stats[i]["score"], reverse=True)
    if requested:
        valid = valid[:min(int(requested), len(valid))]
    return sorted(valid)


def _target_hw_from_config(config):
    target_h, target_w = 112, 96
    input_shape = getattr(config, "input_shape", None)
    if not input_shape:
        return target_h, target_w

    first = str(input_shape).split(";")[0]
    dims = tuple(int(x) for x in first.split(","))
    if len(dims) == 4:
        return int(dims[1]), int(dims[2])
    if len(dims) == 3:
        return int(dims[0]), int(dims[1])
    return target_h, target_w


def _resize_stack(stack, target_h, target_w):
    if stack.shape[1:] == (target_h, target_w):
        return stack
    stack_hwc = np.transpose(stack, (1, 2, 0))
    stack_hwc = cv2.resize(stack_hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.transpose(stack_hwc, (2, 0, 1))


def _load_inputs_loader(files, config, target_h, target_w, norm_counts):
    loader = []
    for path in files:
        combined = np.load(path).astype(np.float32)
        mv = _normalize_stack_to_contract(combined[0:7], norm_counts)
        fx = _normalize_stack_to_contract(combined[7:14], norm_counts)
        mv = _resize_stack(_pad_stack(mv), target_h, target_w)
        fx = _resize_stack(_pad_stack(fx), target_h, target_w)
        loader.append(torch.cat([
            torch.from_numpy(mv).unsqueeze(0),
            torch.from_numpy(fx).unsqueeze(0),
        ], dim=1).permute(0, 2, 3, 1))
    return loader


def _load_stacks_loader(moving_files, fixed_files, target_h, target_w, norm_counts):
    loader = []
    for mv_path, fx_path in zip(moving_files, fixed_files):
        mv = _normalize_stack_to_contract(np.load(mv_path).astype(np.float32), norm_counts)
        fx = _normalize_stack_to_contract(np.load(fx_path).astype(np.float32), norm_counts)
        mv = _resize_stack(_pad_stack(mv), target_h, target_w)
        fx = _resize_stack(_pad_stack(fx), target_h, target_w)
        loader.append(torch.cat([
            torch.from_numpy(mv).unsqueeze(0),
            torch.from_numpy(fx).unsqueeze(0),
        ], dim=1).permute(0, 2, 3, 1))
    return loader


def get_calib_loader(config):
    calib_dir = Path(config.calib_dir)
    stacks_dir = calib_dir / "stacks"
    inputs_dir = calib_dir / "inputs"
    target_h, target_w = _target_hw_from_config(config)
    print(f"[VXM-2P5D-EXPORT] Using target size: {target_h}x{target_w}")

    norm_counts = {
        "already_-1_1": 0,
        "zero_one_to_minus_one_one": 0,
        "minmax_to_minus_one_one": 0,
        "constant_zero": 0,
    }

    moving_files = sorted(stacks_dir.glob("moving_stack_*.npy"))
    fixed_files = sorted(stacks_dir.glob("fixed_stack_*.npy"))

    if moving_files and fixed_files:
        stats = []
        for mv_path, fx_path in zip(moving_files, fixed_files):
            dyn, var, score = _pair_score(
                np.load(mv_path).astype(np.float32),
                np.load(fx_path).astype(np.float32),
            )
            stats.append({"dyn": dyn, "var": var, "score": score})

        selected = _select_indices(stats, config.calib_samples)
        moving_files = [moving_files[i] for i in selected]
        fixed_files = [fixed_files[i] for i in selected]
        print(f"[VXM-2P5D-EXPORT] Selected {len(moving_files)}/{len(stats)} moving/fixed stack pairs")
        loader = _load_stacks_loader(moving_files, fixed_files, target_h, target_w, norm_counts)
        print(f"[VXM-2P5D-EXPORT] Normalization modes: {norm_counts}")
        return loader

    if not inputs_dir.exists():
        raise RuntimeError(f"No calibration data found in {stacks_dir} or {inputs_dir}")

    files = sorted(inputs_dir.glob("input_*.npy"))
    stats = []
    for path in files:
        combined = np.load(path).astype(np.float32)
        dyn, var, score = _pair_score(combined[0:7], combined[7:14])
        stats.append({"dyn": dyn, "var": var, "score": score})

    selected = _select_indices(stats, config.calib_samples)
    files = [files[i] for i in selected]
    print(f"[VXM-2P5D-EXPORT] Selected {len(files)}/{len(stats)} input_* samples")
    loader = _load_inputs_loader(files, config, target_h, target_w, norm_counts)
    print(f"[VXM-2P5D-EXPORT] Normalization modes: {norm_counts}")
    return loader
