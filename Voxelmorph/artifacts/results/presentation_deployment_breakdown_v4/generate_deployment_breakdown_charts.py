"""
Generate deployment breakdown graphics for VoxelMorph 2.5D across FP32 and INT8 precisions.
Uses the exact record construction and styling from generate_v4_precision_comparison.py.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR

BOARD_PATH = RESULTS_ROOT / "inference_pipeline_breakdown" / "inference_pipeline_power_latency_v4.json"
HOST_PATH = RESULTS_ROOT / "inference_only_local_power_latency" / "v4_fp32_int8_host_comparison.json"
GPU_INT8_PATH = RESULTS_ROOT / "inference_only_local_power_latency" / "v4_gpu_int8_tensorrt.json"
ARM_INT8_PATH = RESULTS_ROOT / "int8_arm_cpu_v4_results" / "int8_arm_cpu_results_v4.json"

PLATFORMS = ["FPGA DPU", "ARM CPU", "Local CPU", "Local GPU"]
PRECISIONS = ["FP32", "INT8"]
COLORS = {"FP32": "#3977B8", "INT8": "#E57A2D"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_record(platform: str, precision: str, model: dict, latency_key: str, backend: str, power_scope: str) -> dict:
    latency_ms = float(model[latency_key])
    power_w = float(model["power_mean_w_mean"])
    throughput = 1000.0 / latency_ms

    # Model execution vs data movement
    model_ms = float(model.get("model_inference_ms_mean", 0.0))
    if model_ms == 0.0 and "cpu_model_ms_mean" in model:
        model_ms = float(model["cpu_model_ms_mean"])
    data_ms = max(0.0, latency_ms - model_ms)

    dyn_eng = float(model.get("dynamic_energy_j_per_inference_mean", 0.0))
    dyn_eng_sd = float(model.get("dynamic_energy_j_per_inference_sd", 0.0))

    return {
        "platform": platform,
        "precision": precision,
        "status": "measured",
        "latency_ms_mean": latency_ms,
        "latency_ms_sd": float(model.get(latency_key.replace("_mean", "_sd"), 0.0)),
        "total_s": latency_ms / 1000.0,
        "total_s_sd": float(model.get(latency_key.replace("_mean", "_sd"), 0.0)) / 1000.0,
        "model_s": model_ms / 1000.0,
        "model_s_sd": float(model.get("model_inference_ms_sd", 0.0)) / 1000.0,
        "data_s": data_ms / 1000.0,
        "raw_mean_power_w": power_w,
        "raw_mean_power_sd_w": float(model.get("power_mean_w_sd", 0.0)),
        "raw_energy_j_per_inference": float(model["energy_j_per_inference_mean"]),
        "raw_energy_j_per_inference_sd": float(model.get("energy_j_per_inference_sd", 0.0)),
        "dynamic_energy_j_per_inference": dyn_eng,
        "dynamic_energy_j_per_inference_sd": dyn_eng_sd,
        "throughput_inferences_per_s": throughput,
        "performance_per_watt_inferences_per_s_per_w": throughput / power_w,
        "backend": backend,
        "power_scope": power_scope,
    }


def build_records() -> list[dict]:
    board = read_json(BOARD_PATH)["models"]
    host = read_json(HOST_PATH)["models"]
    gpu_int8 = read_json(GPU_INT8_PATH)["models"]["gpu_int8"]
    arm_int8 = read_json(ARM_INT8_PATH)["models"]["2.5d_fused_arm_cpu"]

    board_scope = "monitored PSINTFP + INT rails"
    desktop_scope = "CPU PPT/package + NVIDIA GPU board"

    records = [
        make_record("FPGA DPU", "INT8", board["2.5d_fused_dpu"], "total_runtime_ms_mean", "Vitis AI 2.5 DPU", board_scope),
        make_record("ARM CPU", "FP32", board["2.5d_fused_arm_cpu"], "total_runtime_ms_mean", "PyTorch FP32", board_scope),
        make_record("ARM CPU", "INT8", arm_int8, "total_runtime_ms_mean", "ONNX Runtime INT8 Quantized", board_scope),
        make_record("Local CPU", "FP32", host["cpu_fp32"], "latency_ms_mean", "PyTorch FP32", desktop_scope),
        make_record("Local CPU", "INT8", host["cpu_int8"], "latency_ms_mean", "ONNX Runtime QOperator INT8", desktop_scope),
        make_record("Local GPU", "FP32", host["gpu_fp32"], "latency_ms_mean", "PyTorch FP32 CUDA", desktop_scope),
        make_record("Local GPU", "INT8", gpu_int8, "latency_ms_mean", "TensorRT INT8 from Vitis Q/DQ ONNX", desktop_scope),
    ]
    return records


def grouped_values(records: list[dict], field: str):
    values = np.full((len(PRECISIONS), len(PLATFORMS)), np.nan)
    errors = np.full_like(values, np.nan)
    error_field = {
        "total_s": "total_s_sd",
        "model_s": "model_s_sd",
        "raw_mean_power_w": "raw_mean_power_sd_w",
        "raw_energy_j_per_inference": "raw_energy_j_per_inference_sd",
        "dynamic_energy_j_per_inference": "dynamic_energy_j_per_inference_sd",
    }.get(field)

    for record in records:
        if record.get("status") != "measured":
            continue
        i = PRECISIONS.index(record["precision"])
        j = PLATFORMS.index(record["platform"])
        values[i, j] = record[field]
        if error_field and error_field in record:
            errors[i, j] = record[error_field]

    return PLATFORMS, PRECISIONS, values, errors


def label_bars(axis, containers, fmt: str) -> None:
    for container in containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append("" if not np.isfinite(height) or height == 0 else format(height, fmt))
        axis.bar_label(container, labels=labels, padding=3, fontsize=8)


def plot_panel(axis, records, field, title, ylabel, log_scale=False, fmt=".3g"):
    platforms, precisions, values, errors = grouped_values(records, field)
    x = np.arange(len(platforms))
    width = 0.34
    containers = []

    for index, precision in enumerate(precisions):
        positions = x + (index - 0.5) * width
        yerr = errors[index] if np.isfinite(errors[index]).any() else None
        container = axis.bar(
            positions,
            values[index],
            width,
            label=precision,
            color=COLORS[precision],
            yerr=yerr,
            capsize=3,
            edgecolor="white",
            linewidth=0.8,
        )
        containers.append(container)

    axis.set_title(title, loc="left", fontsize=13, fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, platforms)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)

    if log_scale:
        axis.set_yscale("log")
        ymin, ymax = axis.get_ylim()
        axis.set_ylim(ymin, ymax * 2.2)
    else:
        ymin, ymax = axis.get_ylim()
        axis.set_ylim(0, ymax * 1.18)

    label_bars(axis, containers, fmt)
    return containers


def save_single(records, field, title, ylabel, filename, log_scale=False, legend_loc="upper right", fmt=".3g", footnote="9 paired volumes • mean ± SD • raw power is not idle-subtracted"):
    figure, axis = plt.subplots(figsize=(11, 5.8))
    containers = plot_panel(axis, records, field, title, ylabel, log_scale=log_scale, fmt=fmt)
    axis.legend(
        handles=[containers[0], containers[1]],
        labels=["FP32", "INT8"],
        frameon=False,
        ncol=2,
        loc=legend_loc,
    )
    figure.text(
        0.01,
        0.01,
        footnote,
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    svg_filename = filename + ".svg"
    png_filename = filename + ".png"
    figure.savefig(OUTPUT_DIR / svg_filename, format="svg", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / png_filename, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_filename} and {svg_filename}")


def plot_stacked_chart(records, filename="stacked_runtime_breakdown"):
    figure, axis = plt.subplots(figsize=(11, 6.2))
    platforms, precisions, model_vals, _ = grouped_values(records, "model_s")
    _, _, data_vals, _ = grouped_values(records, "data_s")

    x = np.arange(len(platforms))
    width = 0.34

    for index, precision in enumerate(precisions):
        positions = x + (index - 0.5) * width
        mod_y = model_vals[index]
        dat_y = data_vals[index]

        # Bottom layer: Model execution
        c1 = axis.bar(
            positions,
            mod_y,
            width,
            color=COLORS[precision],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )
        # Top layer: Data movement + warping (hatched / lighter)
        c2 = axis.bar(
            positions,
            dat_y,
            width,
            bottom=mod_y,
            color=COLORS[precision],
            edgecolor="white",
            linewidth=0.8,
            hatch="//",
            alpha=0.55,
        )

        for p, m_val, d_val in zip(positions, mod_y, dat_y):
            if np.isfinite(m_val) and np.isfinite(d_val):
                tot = m_val + d_val
                axis.text(p, tot * 1.08, f"{tot:.3g}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

    axis.set_title("Runtime breakdown: Model Execution vs. Data Movement + Warping", loc="left", fontsize=13, fontweight="bold")
    axis.set_ylabel("Seconds (log scale)")
    axis.set_xticks(x, platforms)
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)

    ymin, ymax = axis.get_ylim()
    axis.set_ylim(ymin, ymax * 2.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3977B8", edgecolor="white", label="FP32 Model Exec"),
        Patch(facecolor="#3977B8", edgecolor="white", hatch="//", alpha=0.55, label="FP32 Data Move & Warp"),
        Patch(facecolor="#E57A2D", edgecolor="white", label="INT8 Model Exec"),
        Patch(facecolor="#E57A2D", edgecolor="white", hatch="//", alpha=0.55, label="INT8 Data Move & Warp"),
    ]
    axis.legend(handles=legend_elements, frameon=False, ncol=2, loc="upper right", fontsize=9)

    figure.text(
        0.01,
        0.01,
        "9 paired volumes • mean ± SD • Solid = Neural Net Forward Pass, Hatched = Data Movement + Warping",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    svg_filename = filename + ".svg"
    png_filename = filename + ".png"
    figure.savefig(OUTPUT_DIR / svg_filename, format="svg", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / png_filename, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_filename} and {svg_filename}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()

    # Save breakdown json summary
    (OUTPUT_DIR / "deployment_breakdown_summary.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )

    # 1. End-to-end latency
    save_single(
        records,
        "total_s",
        "Inference-to-warp latency",
        "Seconds (log scale)",
        "v4_precision_latency",
        log_scale=True,
    )
    # 2. Data movement + warping latency (Log scale)
    save_single(
        records,
        "data_s",
        "Data movement + warping latency",
        "Seconds (log scale)",
        "data_movement_warp_latency",
        log_scale=True,
    )
    # 3. Model execution latency (Log scale)
    save_single(
        records,
        "model_s",
        "Model execution latency",
        "Seconds (log scale)",
        "model_execution_latency",
        log_scale=True,
    )
    # 4. Raw power (Log scale)
    save_single(
        records,
        "raw_mean_power_w",
        "Raw mean active power",
        "Watts (log scale)",
        "v4_precision_raw_power",
        log_scale=True,
    )
    # 5. Raw energy (Log scale)
    save_single(
        records,
        "raw_energy_j_per_inference",
        "Raw energy per inference",
        "Joules (log scale)",
        "v4_precision_raw_energy",
        log_scale=True,
    )
    # 6. Dynamic energy per inference (idle-subtracted) (Log scale)
    save_single(
        records,
        "dynamic_energy_j_per_inference",
        "Dynamic energy per inference",
        "Joules (log scale)",
        "v4_precision_dynamic_energy",
        log_scale=True,
        footnote="9 paired volumes • mean ± SD • idle baseline subtracted",
    )
    # 7. Performance per Watt (Log scale)
    save_single(
        records,
        "performance_per_watt_inferences_per_s_per_w",
        "Performance per watt",
        "Inferences/s/W (log scale)",
        "v4_precision_performance_per_watt",
        log_scale=True,
    )

    # 8. Stacked chart
    plot_stacked_chart(records, "stacked_runtime_breakdown")

    print("All breakdown charts matching generate_v4_precision_comparison style generated cleanly.")


if __name__ == "__main__":
    main()
