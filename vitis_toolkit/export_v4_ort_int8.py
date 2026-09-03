#!/usr/bin/env python3
"""Export V4 FP32 ONNX and a native ONNX Runtime QOperator INT8 model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


class V4CalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_dir: Path, input_name: str, limit: int) -> None:
        inputs_dir = calibration_dir / "inputs"
        files = sorted(inputs_dir.glob("input_*.npy"))
        if limit > 0:
            files = files[:limit]
        if not files:
            raise FileNotFoundError(f"No calibration tensors found in {inputs_dir}")
        self.rows = []
        for path in files:
            value = np.load(path).astype(np.float32)
            if value.shape != (16, 112, 96):
                raise ValueError(f"{path} has unexpected shape {value.shape}")
            self.rows.append({input_name: value.transpose(1, 2, 0)[None, ...]})
        self._iterator = iter(self.rows)

    def get_next(self):
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = iter(self.rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def operator_counts(path: Path) -> dict[str, int]:
    graph = onnx.load(str(path)).graph
    types = sorted({node.op_type for node in graph.node})
    return {
        operator: sum(node.op_type == operator for node in graph.node)
        for operator in types
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    from vxm_2p5d_v4 import load_model_for_export

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "Vxm2p5dV4_fp32.onnx"
    int8_path = output_dir / "Vxm2p5dV4_ort_qoperator_int8.onnx"
    optimized_path = output_dir / "Vxm2p5dV4_ort_qoperator_int8_optimized.onnx"

    model = load_model_for_export(args.checkpoint.resolve()).cpu().eval()
    dummy = torch.zeros((1, 112, 96, 16), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        input_names=["input"],
        output_names=["flow"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input": {0: "batch"},
            "flow": {0: "batch"},
        },
    )

    reader = V4CalibrationReader(
        args.calibration_dir.resolve(),
        "input",
        args.calibration_samples,
    )
    quantize_static(
        str(fp32_path),
        str(int8_path),
        reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(optimized_path)
    session = ort.InferenceSession(
        str(int8_path),
        options,
        providers=["CPUExecutionProvider"],
    )
    output = session.run(None, {"input": np.zeros((1, 112, 96, 16), np.float32)})
    counts = operator_counts(optimized_path)
    if counts.get("QLinearConv", 0) < 17:
        raise RuntimeError(
            f"Expected 17 native QLinearConv operators, found {counts}"
        )

    manifest = {
        "model": "V4 2.5D",
        "precision": "INT8",
        "backend": "ONNX Runtime CPUExecutionProvider QOperator",
        "quantizer": "ONNX Runtime static MinMax PTQ",
        "calibration_samples": len(reader.rows),
        "calibration_source": str(args.calibration_dir.resolve()),
        "input_shape_nhwc": [1, 112, 96, 16],
        "output_shape_nchw": list(output[0].shape),
        "optimized_operator_counts": counts,
        "note": (
            "Native CPU INT8 uses the same architecture and calibration samples "
            "as Vitis PTQ, but backend-specific scales may differ from Vitis NNDCT."
        ),
    }
    (output_dir / "Vxm2p5dV4_ort_qoperator_int8.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
