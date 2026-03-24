#!/usr/bin/env python3
"""
PyTorch -> Vitis AI 2.5 quantize + compile pipeline
"""

import argparse
import importlib
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Any

try:
    import torch
except ImportError:
    torch = None

try:
    from pytorch_nndct.apis import torch_quantizer
except ImportError:
    torch_quantizer = None
except Exception:
    # Handle environment specific quirks (like missing CUDA libs in CPU docker)
    torch_quantizer = None

from .base import Pipeline
from .config import PipelineConfig


def log(msg: str) -> None:
    print(f"[PYTORCH-PIPELINE] {msg}", flush=True)


class PyTorchPipeline(Pipeline):
    """
    PyTorch implementation of Vitis AI pipeline using NNDCT.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    def load_model(self, module_name: str, class_name: str, checkpoint: Path) -> torch.nn.Module:
        log(f"Loading model {module_name}.{class_name} from {checkpoint}")
        module = importlib.import_module(module_name)
        model_cls = getattr(module, class_name)
        model = model_cls()

        state = torch.load(checkpoint, map_location="cpu")
        if hasattr(model, "net") and isinstance(state, dict):
            model.net.load_state_dict(state)
        else:
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            else:
                model.load_state_dict(state)
        
        model.eval()
        return model

    def run_calibration(self, quant_model: torch.nn.Module, dummy_inputs: tuple) -> None:
        """
        Runs calibration. Uses custom_objects_module.get_calib_loader if available.
        """
        if self.config.custom_objects_module:
            mod = importlib.import_module(self.config.custom_objects_module)
            if hasattr(mod, "get_calib_loader"):
                 loader = mod.get_calib_loader(self.config)
                 log("Running calibration with custom loader...")
                 with torch.no_grad():
                     for batch in loader:
                         if isinstance(batch, (list, tuple)):
                             _ = quant_model(*batch)
                         else:
                             _ = quant_model(batch)
                 return

        # Fallback to dummy
        steps = self.config.calib_samples or 8
        log(f"Running dummy calibration for {steps} steps...")
        with torch.no_grad():
            for _ in range(steps):
                _ = quant_model(*dummy_inputs)

    def quantize(self, model: Optional[torch.nn.Module] = None, input_shape=None, **kwargs) -> Path:
        if input_shape is None: input_shape = [(1, 1, 192, 224)]
        if torch_quantizer is None:
            raise RuntimeError("pytorch_nndct not found")
        
        xir_dir = self.config.output_root / "xir"
        xir_dir.mkdir(parents=True, exist_ok=True)

        # 1) Get model
        if model is None:
             if not self.config.float_model_path:
                 raise ValueError("model or config.float_model_path must be provided")
             # Fallback to old loading logic if module/class provided in kwargs
             model = self.load_model(kwargs.get("model_module"), kwargs.get("model_class"), self.config.float_model_path)

        # 2) Dummy input for tracing
        if isinstance(input_shape, list):
            dummy_inputs = tuple(torch.randn(*s) for s in input_shape)
        else:
            dummy_inputs = (torch.randn(*input_shape),)

        # === CALIBRATION PHASE ===
        quantizer = torch_quantizer(
            "calib",
            model,
            dummy_inputs,
            output_dir=str(xir_dir),
        )
        quant_model = quantizer.quant_model

        self.run_calibration(quant_model, dummy_inputs)

        quantizer.export_quant_config()

        # === EXPORT XMODEL ===
        try:
            quantizer.export_xmodel(output_dir=str(xir_dir), deploy_check=False)
        except TypeError:
            quantizer.export_xmodel(deploy_check=False)

        xmodels = sorted(xir_dir.glob("*.xmodel"))
        if not xmodels:
            raise RuntimeError(f"No .xmodel generated in {xir_dir}")

        int_candidates = [p for p in xmodels if p.stem.endswith("_int")]
        return int_candidates[0] if int_candidates else xmodels[0]

    def compile(self, xir_model: Path) -> Path:
        compile_dir = self.config.output_root / "compiled"
        compile_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "vai_c_xir",
            "-x", str(xir_model),
            "-a", str(self.config.arch_json),
            "-o", str(compile_dir),
            "-n", self.config.net_name,
        ]
        if self.config.extra_compiler_args:
             cmd.extend(self.config.extra_compiler_args)

        log("Running compiler:")
        log("  " + " ".join(cmd))
        subprocess.run(cmd, check=True)

        xmodels = sorted(compile_dir.glob("*.xmodel"))
        if not xmodels:
            raise RuntimeError(f"No compiled .xmodel generated in {compile_dir}")

        return xmodels[0]


def parse_shape(shape_str: str):
    if ';' in shape_str:
        return [tuple(int(x) for x in s.split(',')) for s in shape_str.split(';')]
    elif ':' in shape_str:
        return [tuple(int(x) for x in s.split(',')) for s in shape_str.split(':')]
    return [tuple(int(x) for x in shape_str.split(','))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-module", required=True)
    parser.add_argument("--model-class", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--arch-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--net-name", required=True)
    parser.add_argument("--calib-steps", type=int, default=8)

    args = parser.parse_args()
    
    config = PipelineConfig(
        target_dpu="N/A", # Not used directly by NNDCT in 'calib' mode call
        arch_json=args.arch_json,
        output_root=args.output_dir,
        float_model_path=args.checkpoint,
        net_name=args.net_name,
        calib_samples=args.calib_steps
    )

    pipeline = PyTorchPipeline(config)
    xmodel = pipeline.run(
        model_module=args.model_module,
        model_class=args.model_class,
        input_shape=parse_shape(args.input_shape)
    )
    log(f"Pipeline complete: {xmodel}")


if __name__ == "__main__":
    main()

