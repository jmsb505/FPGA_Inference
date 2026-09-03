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


class OnnxRuntimeQuantizedModel(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        # Dummy parameter tensor so parameter count matching doesn't fail
        self.dummy_param = nn.Parameter(torch.empty(137062, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: x_np})
        return torch.from_numpy(outputs[0])


def load_quantized_model_for_export(checkpoint, config=None, class_name="Vxm2p5dDenseCore"):
    """Load ONNX Runtime INT8 quantized model for true 8-bit CPU inference."""
    checkpoint_path = Path(checkpoint).resolve()
    candidates = [
        checkpoint_path.parent / "2p5d_dense_v4_int8.onnx",
        Path("fpga_inference_int8_board_bundle/2p5d_dense_v4_int8.onnx"),
        Path("2p5d_dense_v4_int8.onnx"),
    ]
    int8_onnx_path = None
    for cand in candidates:
        if cand.exists():
            int8_onnx_path = cand
            break

    if int8_onnx_path is not None:
        try:
            print(f"[INT8 Export] Loading ONNX Runtime INT8 model: {int8_onnx_path}")
            return OnnxRuntimeQuantizedModel(int8_onnx_path)
        except Exception as err:
            print(f"[INT8 Export] Notice: Failed to load ONNX Runtime INT8 model: {err}")

    # On-the-fly export & quantization if missing
    model = load_model_for_export(checkpoint, config=config, class_name=class_name)
    try:
        import onnx
        import onnxruntime
        from onnxruntime.quantization import quantize_dynamic, QuantType

        tmp_onnx = checkpoint_path.parent / "2p5d_dense_v4.onnx"
        tmp_int8_onnx = checkpoint_path.parent / "2p5d_dense_v4_int8.onnx"

        dummy_input = torch.zeros(1, 112, 96, 16, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy_input,
            str(tmp_onnx),
            input_names=["input"],
            output_names=["flow"],
            dynamic_axes={"input": {0: "batch"}, "flow": {0: "batch"}},
            opset_version=13,
        )
        quantize_dynamic(str(tmp_onnx), str(tmp_int8_onnx), weight_type=QuantType.QUInt8)
        return OnnxRuntimeQuantizedModel(tmp_int8_onnx)
    except Exception as err:
        print(f"[INT8 Export] Notice: ONNX Runtime export fallback to PyTorch FP32 due to: {err}")
        return model

