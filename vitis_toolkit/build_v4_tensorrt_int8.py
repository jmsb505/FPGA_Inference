#!/usr/bin/env python3
"""Build and audit a TensorRT engine from the Vitis Q/DQ V4 ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--max-batch", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = args.onnx.resolve().read_bytes()
    if not parser.parse(onnx_bytes):
        errors = [
            str(parser.get_error(index)) for index in range(parser.num_errors)
        ]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_gib * 1024**3),
    )
    config.set_flag(trt.BuilderFlag.INT8)
    config.clear_flag(trt.BuilderFlag.TF32)
    input_tensor = network.get_input(0)
    input_shape = tuple(input_tensor.shape)
    if args.max_batch > 1:
        input_tensor.shape = (-1, *input_shape[1:])
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            (1, *input_shape[1:]),
            (args.max_batch, *input_shape[1:]),
            (args.max_batch, *input_shape[1:]),
        )
        config.add_optimization_profile(profile)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the INT8 engine")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the built engine")

    inspector = engine.create_engine_inspector()
    layer_json = inspector.get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    layer_path = args.engine.with_suffix(".layers.json")
    layer_path.write_text(layer_json + "\n", encoding="utf-8")
    lower = layer_json.lower()
    int8_mentions = lower.count("int8")
    fp16_mentions = lower.count("fp16")
    fp32_mentions = lower.count("fp32") + lower.count("float")
    if int8_mentions == 0:
        raise RuntimeError(
            "Built engine contains no INT8 layer metadata; refusing to label "
            "this result as native INT8"
        )

    bindings = []
    for index in range(engine.num_bindings):
        bindings.append(
            {
                "index": index,
                "name": engine.get_binding_name(index),
                "is_input": bool(engine.binding_is_input(index)),
                "dtype": str(engine.get_binding_dtype(index)),
                "shape": list(engine.get_binding_shape(index)),
            }
        )
    manifest = {
        "source_onnx": str(args.onnx.resolve()),
        "engine": str(args.engine.resolve()),
        "tensorrt": trt.__version__,
        "builder_int8_flag": True,
        "tf32_disabled": True,
        "max_batch": args.max_batch,
        "int8_mentions_in_layer_audit": int8_mentions,
        "fp16_mentions_in_layer_audit": fp16_mentions,
        "fp32_or_float_mentions_in_layer_audit": fp32_mentions,
        "bindings": bindings,
        "layer_audit": str(layer_path.resolve()),
    }
    manifest_path = args.engine.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
