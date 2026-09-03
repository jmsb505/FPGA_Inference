"""
Generate deployment breakdown graphics for VoxelMorph 2.5D across FP32 and INT8 precisions.
Uses the exact visual style, colors, despined axes, bar labels, and typography from generate_v4_precision_comparison.py.
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

# Source result files
LOCAL_FP32_PATH = RESULTS_ROOT / "inference_only_local_power_latency" / "local_2p5d_3d_cpu_gpu_9_pairs.json"
LOCAL_CPU_INT8_PATH = RESULTS_ROOT / "inference_only_local_power_latency" / "v4_fp32_int8_host_comparison.json"
LOCAL_GPU_INT8_PATH = RESULTS_ROOT / "inference_only_local_power_latency" / "v4_gpu_int8_tensorrt.json"
BOARD_FP32_DPU_PATH = RESULTS_ROOT / "inference_pipeline_breakdown" / "fpga_board_9_pairs.json"
BOARD_ARM_INT8_PATH = RESULTS_ROOT / "int8_arm_cpu_v4_results" / "int8_arm_cpu_results_v4.json"

PLATFORMS = ["FPGA DPU", "ARM CPU", "Local CPU", "Local GPU"]
PRECISIONS = ["FP32", "INT8"]
COLORS = {"FP32": "#3977B8", "INT8": "#E57A2D"}

def build_records() -> list[dict]:
    def _load(p):
        return json.loads(p.read_text(encoding="utf-8"))

    local_fp32 = _load(LOCAL_FP32_PATH)
    local_cpu_int8 = _load(LOCAL_CPU_INT8_PATH)
    local_gpu_int8 = _load(LOCAL_GPU_INT8_PATH)
    board_fp32_dpu = _load(BOARD_FP32_DPU_PATH)
    board_arm_int8 = _load(BOARD_ARM_INT8_PATH)

    def make_rec(platform, precision, data_dict, backend):
        tot_ms = float(data_dict.get("total_runtime_ms_mean") or data_dict.get("latency_ms_mean") or 0.0)
        tot_sd = float(data_dict.get("total_runtime_ms_sd") or data_dict.get("latency_ms_sd") or 0.0)
        mod_ms = float(data_dict.get("model_inference_ms_mean") or 0.0)
        mod_sd = float(data_dict.get("model_inference_ms_sd") or 0.0)
        data_ms = max(0.0, tot_ms - mod_ms)

        pwr = float(data_dict.get("power_mean_w_mean") or data_dict.get("power_mean_w") or data_dict.get("raw_mean_power_w") or 0.0)
        pwr_sd = float(data_dict.get("power_mean_w_sd") or data_dict.get("raw_mean_power_sd_w") or 0.0)
        eng = float(data_dict.get("energy_j_per_inference_mean") or data_dict.get("raw_energy_j") or data_dict.get("raw_energy_j_per_inference") or 0.0)
        eng_sd = float(data_dict.get("energy_j_per_inference_sd") or data_dict.get("raw_energy_j_per_inference_sd") or 0.0)

        tp = float(data_dict.get("throughput_inferences_per_s") or (1000.0 / tot_ms if tot_ms > 0 else 0.0))
        pw = float(data_dict.get("performance_per_watt_inferences_per_s_per_w") or (tp / pwr if pwr > 0 else 0.0))

        return {
            "platform": platform,
            "precision": precision,
            "backend": backend,
            "status": "measured",
            "total_s": tot_ms / 1000.0,
            "total_s_sd": tot_sd / 1000.0,
            "model_s": mod_ms / 1000.0,
            "model_s_sd": mod_sd / 1000.0,
            "data_s": data_ms / 1000.0,
            "raw_mean_power_w": pwr,
            "raw_mean_power_sd_w": pwr_sd,
            "raw_energy_j_per_inference": eng,
            "raw_energy_j_per_inference_sd": eng_sd,
            "throughput_inferences_per_s": tp,
            "performance_per_watt_inferences_per_s_per_w": pw,
        }

    records = [
        make_rec("Local CPU", "FP32", local_fp32["models"]["2.5d_cpu"], "PyTorch FP32"),
        make_rec("Local CPU", "INT8", local_cpu_int8["models"]["cpu_int8"], "ONNX Runtime QOperator INT8"),
        make_rec("Local GPU", "FP32", local_fp32["models"]["2.5d_gpu"], "PyTorch FP32 CUDA"),
        make_rec("Local GPU", "INT8", local_gpu_int8["models"]["gpu_int8"], "TensorRT INT8 from Vitis Q/DQ ONNX"),
        make_rec("ARM CPU", "FP32", board_fp32_dpu["models"]["2.5d_fused_arm_cpu"], "PyTorch FP32"),
        make_rec("ARM CPU", "INT8", board_arm_int8["models"]["2.5d_fused_arm_cpu"], "ONNX Runtime INT8 Quantized"),
        make_rec("FPGA DPU", "INT8", board_fp32_dpu["models"]["2.5d_fused_dpu"], "Vitis AI 2.5 DPU"),
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
            labels.append("" if not np.isfinite(height) else format(height, fmt))
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
        axis.set_ylim(ymin, ymax * 1.15)

    label_bars(axis, containers, fmt)
    return containers


def save_single(records, field, title, ylabel, filename, log_scale=False, legend_loc="upper right", fmt=".3g"):
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
        "9 paired volumes • mean ± SD • raw power is not idle-subtracted",
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
            alpha=0.5,
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

    # Custom legend for FP32/INT8 + Model vs Data Movement
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3977B8", edgecolor="white", label="FP32 Model Exec"),
        Patch(facecolor="#3977B8", edgecolor="white", hatch="//", alpha=0.5, label="FP32 Data Move & Warp"),
        Patch(facecolor="#E57A2D", edgecolor="white", label="INT8 Model Exec"),
        Patch(facecolor="#E57A2D", edgecolor="white", hatch="//", alpha=0.5, label="INT8 Data Move & Warp"),
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

    # Generate all single figures matching style of generate_v4_precision_comparison.py
    save_single(
        records,
        "total_s",
        "Inference-to-warp latency",
        "Seconds (log scale)",
        "v4_precision_latency",
        log_scale=True,
    )
    save_single(
        records,
        "data_s",
        "Data movement + warping latency",
        "Seconds (log scale)",
        "data_movement_warp_latency",
        log_scale=True,
    )
    save_single(
        records,
        "model_s",
        "Model execution latency",
        "Seconds (log scale)",
        "model_execution_latency",
        log_scale=True,
    )
    save_single(
        records,
        "raw_mean_power_w",
        "Raw mean active power",
        "Watts (log scale)",
        "v4_precision_raw_power",
        log_scale=True,
    )
    save_single(
        records,
        "raw_energy_j_per_inference",
        "Raw energy per inference",
        "Joules (log scale)",
        "v4_precision_raw_energy",
        log_scale=True,
    )
    save_single(
        records,
        "performance_per_watt_inferences_per_s_per_w",
        "Performance per watt",
        "Inferences/s/W (log scale)",
        "v4_precision_performance_per_watt",
        log_scale=True,
    )

    # Stacked chart
    plot_stacked_chart(records, "stacked_runtime_breakdown")

    print("All breakdown charts matching generate_v4_precision_comparison style generated cleanly.")


if __name__ == "__main__":
    main()
