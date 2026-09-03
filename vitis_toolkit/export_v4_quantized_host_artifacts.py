#!/usr/bin/env python3
"""Rebuild the V4 NNDCT test graph and export host-side INT8 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from pytorch_nndct.apis import torch_quantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quant-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    from vxm_2p5d_v4 import load_model_for_export

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    quant_dir = args.quant_dir.resolve()
    quant_info = quant_dir / "quant_info.json"
    if not quant_info.exists():
        raise FileNotFoundError(f"Missing Vitis quantization metadata: {quant_info}")

    model = load_model_for_export(args.checkpoint.resolve()).cpu().eval()
    dummy = torch.zeros((1, 112, 96, 16), dtype=torch.float32)
    quantizer = torch_quantizer(
        "test",
        model,
        (dummy,),
        output_dir=str(quant_dir),
        device=torch.device("cpu"),
    )
    quant_model = quantizer.quant_model.cpu().eval()
    with torch.no_grad():
        output = quant_model(dummy)

    torch.save(quant_model.state_dict(), output_dir / "v4_nndct_test_state_dict.pth")
    (output_dir / "v4_nndct_test_graph.txt").write_text(
        repr(quant_model), encoding="utf-8"
    )

    export_error = None
    try:
        quantizer.export_onnx_model(output_dir=str(output_dir), verbose=True)
    except Exception as exc:  # Preserve the exact Vitis error in the manifest.
        export_error = f"{type(exc).__name__}: {exc}"

    onnx_files = sorted(output_dir.glob("*.onnx"))
    operator_types: list[str] = []
    onnx_error = None
    if onnx_files:
        try:
            import onnx

            graph = onnx.load(str(onnx_files[0])).graph
            operator_types = sorted({node.op_type for node in graph.node})
        except Exception as exc:
            onnx_error = f"{type(exc).__name__}: {exc}"

    manifest = {
        "quantization": "Vitis AI 2.5 NNDCT post-training INT8 test graph",
        "execution_semantics": (
            "Host NNDCT test graph reproduces Vitis quantized numerics; native "
            "INT8 acceleration must be verified separately for each backend."
        ),
        "input_shape_nhwc": [1, 112, 96, 16],
        "output_shape_nchw": list(output.shape),
        "quant_info": str(quant_info),
        "onnx_files": [path.name for path in onnx_files],
        "onnx_operator_types": operator_types,
        "onnx_export_error": export_error,
        "onnx_inspection_error": onnx_error,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
