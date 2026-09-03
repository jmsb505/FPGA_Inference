"""Build the drop-in FP32 ARM-CPU notebook bundle for the FPGA board."""

from __future__ import annotations

import hashlib
import json
import shutil
import textwrap
import zipfile
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
VXM = ROOT / "Voxelmorph"
SOURCE = VXM / "fpga_inference_v4.ipynb"
NOTEBOOK = VXM / "fpga_inference_fp32.ipynb"
MODEL = VXM / "trained_weights" / "2p5d_dense_pt_v4_canonical_best.pth"
MODEL_HELPER = ROOT / "vxm_2p5d_export.py"
TIMING_HELPER = VXM / "fpga_deployment_timing.py"
POWER_HELPER = VXM / "fpga_deployment_power.py"
BUNDLE_DIR = VXM / "fpga_inference_fp32_board_bundle"
BUNDLE_ZIP = VXM / "fpga_inference_fp32_board_bundle.zip"


def clean(source):
    return textwrap.dedent(source).strip() + "\n"


def source_cell(notebook, marker):
    for cell in notebook.cells:
        if cell.cell_type == "code" and marker in cell.source:
            return cell.source
    raise RuntimeError(f"Missing source cell marker: {marker}")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build_notebook():
    current = nbformat.read(SOURCE, as_version=4)
    geometry = source_cell(current, "def normalize_volume_contract")
    geometry = geometry.split("def _run_int8_dpu", 1)[0].rstrip()
    runtime = source_cell(current, "def compute_dice_per_label")
    old = (
        "flow, elapsed = run_dpu_inference(moving_stack, fixed_stack) "
        "if device == 'fpga' else run_cpu_inference(moving_stack, "
        "fixed_stack, weights_path)"
    )
    new = "flow, elapsed = run_cpu_inference(moving_stack, fixed_stack, weights_path)"
    if old not in runtime:
        raise RuntimeError("Current notebook CPU dispatch was not found")
    runtime = runtime.replace(old, new)

    cells = [
        nbformat.v4.new_markdown_cell(clean(
            """
            # V4 2.5D FP32 inference on the FPGA board

            Runs the non-quantized FP32 model on the board ARM CPU. The DPU
            graph is INT8-only, so this is the valid FP32 board baseline.
            Existing NPY test volumes are reused from this directory or
            ./data/test_data/. The V4 canonical grid, orientation-aware
            letterbox, three-orientation mean fusion, and inference-to-warp
            measurement boundary are unchanged.
            """
        )),
        nbformat.v4.new_code_cell(clean(
            """
            import json
            import os
            from pathlib import Path
            import cv2
            import matplotlib.pyplot as plt
            import numpy as np
            import pynq
            import torch
            import torch.nn.functional as F
            from scipy.ndimage import gaussian_filter, map_coordinates

            PIPELINE_VERSION = "v4-canonical-letterbox-fp32-arm"
            WINDOW_RADIUS = 3
            N_STACK = 8
            INPUT_HEIGHT, INPUT_WIDTH = 112, 96
            INPUT_CHANNELS = 16
            CANONICAL_VOLUME_SHAPE = (96, 112, 96)
            PAD_VALUE = -1.0
            SEG_LABELS = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28]
            CPU_EXPORT_HELPER = "./vxm_2p5d_export.py"
            WEIGHTS_PATH = "./2p5d_dense_pt_v4_canonical_best.pth"
            OUTPUT_PATH = "fp32_arm_board_results_v4.json"
            PAIR_LIMIT = None
            LATENCY_REPETITIONS = 3
            MINIMUM_POWER_WINDOW_S = 10.0
            POWER_SAMPLE_INTERVAL_S = 0.10
            IDLE_CALIBRATION_S = 20.0

            def resolve_file(default_path, extra_candidates=None):
                candidates = [Path(default_path)]
                candidates.extend(Path(item) for item in (extra_candidates or []))
                for candidate in candidates:
                    if candidate.exists():
                        return str(candidate)
                return str(candidates[0])

            for required in (CPU_EXPORT_HELPER, WEIGHTS_PATH):
                if not Path(required).exists():
                    raise FileNotFoundError(
                        f"Missing {required}; extract the complete bundle here."
                    )
            print("Target: FPGA-board ARM CPU")
            print("Precision: FP32 (non-quantized)")
            print("Power scope: monitored PSINTFP + INT rails")
            """
        )),
        nbformat.v4.new_markdown_cell(
            "## Validated V4 preprocessing and runtime\n\n"
            "Derived from the current FPGA inference notebook."
        ),
        nbformat.v4.new_code_cell(geometry),
        nbformat.v4.new_code_cell(runtime),
        nbformat.v4.new_code_cell(clean(
            """
            helper_dir = str(Path(CPU_EXPORT_HELPER).resolve().parent)
            if helper_dir not in os.sys.path:
                os.sys.path.insert(0, helper_dir)
            from vxm_2p5d_export import load_model_for_export

            MODEL_CPU = load_model_for_export(WEIGHTS_PATH).to("cpu").eval()
            PARAMETER_COUNT = sum(p.numel() for p in MODEL_CPU.parameters())
            with torch.inference_mode():
                output = MODEL_CPU(torch.zeros(
                    1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS
                ))
            assert tuple(output.shape) == (1, 2, INPUT_HEIGHT, INPUT_WIDTH)
            print(f"Model ready: {PARAMETER_COUNT:,} parameters")
            """
        )),
        nbformat.v4.new_markdown_cell(clean(
            """
            ## FP32 ARM benchmark

            Latency is unmonitored. A separate identical execution records raw
            power and energy. Raw power is not idle-subtracted and is not
            whole-system wall power.
            """
        )),
        nbformat.v4.new_code_cell(clean(
            """
            from fpga_deployment_timing import PAIRS
            from fpga_deployment_power import run_benchmark

            selected_pairs = PAIRS if PAIR_LIMIT is None else PAIRS[:PAIR_LIMIT]
            results = run_benchmark(
                globals(),
                output_path=OUTPUT_PATH,
                latency_repetitions=LATENCY_REPETITIONS,
                minimum_power_window_s=MINIMUM_POWER_WINDOW_S,
                power_sample_interval_s=POWER_SAMPLE_INTERVAL_S,
                idle_calibration_s=IDLE_CALIBRATION_S,
                devices=("cpu",),
                pairs=selected_pairs,
            )
            arm = results["models"]["2.5d_fused_arm_cpu"]
            throughput = 1000.0 / arm["total_runtime_ms_mean"]
            performance_per_watt = throughput / arm["power_mean_w_mean"]
            arm["throughput_inferences_per_s"] = throughput
            arm["performance_per_watt_inferences_per_s_per_w"] = performance_per_watt
            results["execution"] = {
                "target": "FPGA board ARM CPU",
                "precision": "FP32",
                "quantized": False,
                "accelerator": "none",
                "fusion": "mean",
                "parameter_count": PARAMETER_COUNT,
            }
            Path(OUTPUT_PATH).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Latency: {arm['total_runtime_ms_mean'] / 1000.0:.3f} s")
            print(f"Raw mean power: {arm['power_mean_w_mean']:.3f} W")
            print(f"Raw energy: {arm['energy_j_per_inference_mean']:.3f} J/inference")
            print(f"Performance/W: {performance_per_watt:.5f} inf/s/W")
            """
        )),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)


def build_bundle():
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir()
    sources = (NOTEBOOK, MODEL, MODEL_HELPER, TIMING_HELPER, POWER_HELPER)
    for source in sources:
        shutil.copy2(source, BUNDLE_DIR / source.name)
    readme = clean(
        """
        FP32 ARM-CPU BOARD BUNDLE

        Extract every file directly into the directory containing the current
        FPGA inference notebook. Open fpga_inference_fp32.ipynb and run all
        cells. The existing dataset is reused and is not included.

        This runs FP32 PyTorch on the board ARM CPU. The DPU is INT8-only, so
        this bundle needs no xmodel or bitstream.

        Output: fp32_arm_board_results_v4.json

        The full nine-pair run can take roughly 25-30 minutes. For a smoke
        test set PAIR_LIMIT=1, LATENCY_REPETITIONS=1,
        IDLE_CALIBRATION_S=3.0, and MINIMUM_POWER_WINDOW_S=1.0.
        """
    )
    (BUNDLE_DIR / "README_FP32_BOARD.txt").write_text(
        readme, encoding="utf-8"
    )
    manifest = {
        "archive_layout": "flat; extract beside the current board notebook",
        "dataset_included": False,
        "target": "FPGA board ARM CPU",
        "precision": "FP32",
        "files": {},
    }
    for path in sorted(BUNDLE_DIR.iterdir()):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    (BUNDLE_DIR / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if BUNDLE_ZIP.exists():
        BUNDLE_ZIP.unlink()
    with zipfile.ZipFile(BUNDLE_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUNDLE_DIR.iterdir()):
            archive.write(path, path.name)


def main():
    required = (SOURCE, MODEL, MODEL_HELPER, TIMING_HELPER, POWER_HELPER)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    build_notebook()
    build_bundle()
    print(f"Notebook: {NOTEBOOK}")
    print(f"Bundle: {BUNDLE_ZIP}")


if __name__ == "__main__":
    main()
