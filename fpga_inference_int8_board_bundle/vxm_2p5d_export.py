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
    new_w[:, :7, :, :] = ckpt_w[:, :7, :, :]
    new_w[:, 8:15, :, :] = ckpt_w[:, 7:14, :, :]
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


def load_quantized_model_for_export(checkpoint, config=None, class_name="Vxm2p5dDenseCore"):
    """Load model and apply INT8 dynamic quantization for CPU inference."""
    model = load_model_for_export(checkpoint, config=config, class_name=class_name)
    try:
        if hasattr(torch.ao, "quantization"):
            quant_fn = torch.ao.quantization.quantize_dynamic
        else:
            quant_fn = torch.quantization.quantize_dynamic
        model_int8 = quant_fn(model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
        model_int8.eval()
        return model_int8
    except Exception as err:
        print(f"Notice: torch dynamic quantization fallback to FP32 weights due to: {err}")
        return model
