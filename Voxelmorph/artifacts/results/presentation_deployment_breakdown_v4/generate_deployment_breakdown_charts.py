"""
Generate comprehensive deployment breakdown graphics for VoxelMorph 2.5D across all 7 platform & precision configurations:
- Local CPU (FP32 & INT8)
- Local GPU (FP32 & INT8)
- ARM CPU (FP32 & INT8)
- FPGA DPU (INT8)

Produces 300 DPI PNGs and SVGs for:
1. Model execution latency
2. Data movement + warping latency
3. Total inference-to-warp latency
4. Stacked runtime breakdown (Model vs Data Movement)
5. Mean power, energy, and performance per watt
6. Individual runtime share pie charts
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

# Paths to canonical result sources
LOCAL_FP32_PATH = RESULTS_ROOT / "inference_only_local_power_latency" / "local_2p5d_3d_cpu_gpu_9_pairs.json"
LOCAL_CPU_INT8_PATH = RESULTS_ROOT / "inference_only_local_power_latency" / "v4_fp32_int8_host_comparison.json"
LOCAL_GPU_INT8_PATH = RESULTS_ROOT / "inference_only_local_power_latency" / "v4_gpu_int8_tensorrt.json"
BOARD_FP32_DPU_PATH = RESULTS_ROOT / "inference_pipeline_breakdown" / "fpga_board_9_pairs.json"
BOARD_ARM_INT8_PATH = RESULTS_ROOT / "int8_arm_cpu_v4_results" / "int8_arm_cpu_results_v4.json"

# Palette definitions matching v4 precision style
PRECISION_COLORS = {
    "FP32": "#D97706",       # Warm Amber/Orange
    "INT8": "#0284C7",       # Modern Ocean Blue
}

PLATFORM_COLORS = {
    ("Local CPU", "FP32"): "#F59E0B",
    ("Local CPU", "INT8"): "#0284C7",
    ("Local GPU", "FP32"): "#10B981",
    ("Local GPU", "INT8"): "#0EA5E9",
    ("ARM CPU", "FP32"):   "#8B5CF6",
    ("ARM CPU", "INT8"):   "#6366F1",
    ("FPGA DPU", "INT8"):  "#1D4ED8",
}

def load_data():
    def _read(p):
        return json.loads(p.read_text(encoding="utf-8"))

    local_fp32 = _read(LOCAL_FP32_PATH)
    local_cpu_int8 = _read(LOCAL_CPU_INT8_PATH)
    local_gpu_int8 = _read(LOCAL_GPU_INT8_PATH)
    board_fp32_dpu = _read(BOARD_FP32_DPU_PATH)
    board_arm_int8 = _read(BOARD_ARM_INT8_PATH)

    def extract_entry(model_dict, platform, precision, backend):
        tot = float(model_dict.get("total_runtime_ms_mean") or model_dict.get("latency_ms_mean") or 0.0)
        tot_sd = float(model_dict.get("total_runtime_ms_sd") or model_dict.get("latency_ms_sd") or 0.0)
        mod = float(model_dict.get("model_inference_ms_mean") or 0.0)
        mod_sd = float(model_dict.get("model_inference_ms_sd") or 0.0)
        data_ms = max(0.0, tot - mod)
        
        pwr = float(model_dict.get("power_mean_w_mean") or model_dict.get("power_mean_w") or model_dict.get("raw_mean_power_w") or 0.0)
        pwr_sd = float(model_dict.get("power_mean_w_sd") or model_dict.get("raw_mean_power_sd_w") or 0.0)
        eng = float(model_dict.get("energy_j_per_inference_mean") or model_dict.get("raw_energy_j") or model_dict.get("raw_energy_j_per_inference") or 0.0)
        eng_sd = float(model_dict.get("energy_j_per_inference_sd") or model_dict.get("raw_energy_j_per_inference_sd") or 0.0)
        dyn_eng = float(model_dict.get("dynamic_energy_j_per_inference_mean") or 0.0)
        dyn_eng_sd = float(model_dict.get("dynamic_energy_j_per_inference_sd") or 0.0)
        
        tp = float(model_dict.get("throughput_inferences_per_s") or (1000.0 / tot if tot > 0 else 0.0))
        pw = float(model_dict.get("performance_per_watt_inferences_per_s_per_w") or (tp / pwr if pwr > 0 else 0.0))

        return {
            "platform": platform,
            "precision": precision,
            "backend": backend,
            "display_name": f"{platform} ({precision})",
            "total_ms": tot,
            "total_ms_sd": tot_sd,
            "model_ms": mod,
            "model_ms_sd": mod_sd,
            "data_ms": data_ms,
            "power_w": pwr,
            "power_w_sd": pwr_sd,
            "energy_j": eng,
            "energy_j_sd": eng_sd,
            "dynamic_energy_j": dyn_eng,
            "dynamic_energy_j_sd": dyn_eng_sd,
            "throughput": tp,
            "perf_per_w": pw,
        }

    records = [
        extract_entry(local_fp32["models"]["2.5d_cpu"], "Local CPU", "FP32", "PyTorch FP32"),
        extract_entry(local_cpu_int8["models"]["cpu_int8"], "Local CPU", "INT8", "ONNX Runtime QOperator"),
        extract_entry(local_fp32["models"]["2.5d_gpu"], "Local GPU", "FP32", "PyTorch CUDA FP32"),
        extract_entry(local_gpu_int8["models"]["gpu_int8"], "Local GPU", "INT8", "TensorRT INT8"),
        extract_entry(board_fp32_dpu["models"]["2.5d_fused_arm_cpu"], "ARM CPU", "FP32", "PyTorch FP32"),
        extract_entry(board_arm_int8["models"]["2.5d_fused_arm_cpu"], "ARM CPU", "INT8", "ONNX Runtime ConvInteger"),
        extract_entry(board_fp32_dpu["models"]["2.5d_fused_dpu"], "FPGA DPU", "INT8", "Vitis AI 2.5 DPU"),
    ]
    return records

def save_fig(fig, name):
    png_path = OUTPUT_DIR / f"{name}.png"
    svg_path = OUTPUT_DIR / f"{name}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path.name} and {svg_path.name}")

def plot_bar_chart(records, val_key, err_key, title, ylabel, filename, log_scale=False, format_str="{:,.1f}"):
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [r["display_name"] for r in records]
    vals = [r[val_key] for r in records]
    errs = [r[err_key] if err_key else 0.0 for r in records]
    colors = [PLATFORM_COLORS[(r["platform"], r["precision"])] for r in records]

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=errs if any(errs) else None, capsize=4, color=colors, edgecolor="#333333", linewidth=0.8, width=0.6, zorder=3)

    ax.set_title(title, fontsize=15, pad=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    if log_scale:
        ax.set_yscale("log")
        max_val = max(vals)
        ax.set_ylim(bottom=max(0.1, min(v for v in vals if v > 0) * 0.5), top=max_val * 2.5)
    else:
        max_val = max(v + e for v, e in zip(vals, errs))
        ax.set_ylim(top=max_val * 1.25)

    for bar, val, err in zip(bars, vals, errs):
        h = bar.get_height()
        txt = format_str.format(val)
        offset = (val * 0.08 if log_scale else max(err, max_val * 0.02)) + 1e-4
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + offset, txt, ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.text(0.5, 0.01, "Matched inference-to-warp boundary across 9 canonical volume pairs.", ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, filename)

def plot_stacked_chart(records, filename):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    labels = [r["display_name"] for r in records]
    model_ms = [r["model_ms"] for r in records]
    data_ms = [r["data_ms"] for r in records]
    totals = [r["total_ms"] for r in records]

    x = np.arange(len(labels))
    width = 0.58

    b1 = ax.bar(x, model_ms, width, label="Model Execution (NN Forward Pass)", color="#0284C7", edgecolor="#333333", zorder=3)
    b2 = ax.bar(x, data_ms, width, bottom=model_ms, label="Data Movement + Warping (Host CPU)", color="#94A3B8", edgecolor="#333333", zorder=3)

    ax.set_title("Inference-to-Warp Runtime Breakdown (Model vs. Data Movement)", fontsize=15, pad=15, fontweight="bold")
    ax.set_ylabel("Milliseconds (log scale)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(bottom=10, top=max(totals) * 3.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    for i, (tot, mod, dat) in enumerate(zip(totals, model_ms, data_ms)):
        pct_mod = (mod / tot) * 100 if tot > 0 else 0
        pct_dat = (dat / tot) * 100 if tot > 0 else 0
        ax.text(x[i], tot * 1.15, f"{tot:,.1f} ms\n({pct_mod:.0f}% NN / {pct_dat:.0f}% Move)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)
    fig.text(0.5, 0.01, "Demonstrates how INT8 reduces NN execution while host data movement overhead remains constant.", ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, filename)

def plot_pie_charts(records):
    for r in records:
        labels = ["Model Execution", "Data Movement + Warping"]
        vals = [r["model_ms"], r["data_ms"]]
        tot = r["total_ms"]

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        wedges, texts, autotexts = ax.pie(
            vals,
            labels=labels,
            colors=["#0284C7", "#94A3B8"],
            startangle=90,
            counterclock=False,
            autopct=lambda pct: f"{pct:.1f}%",
            pctdistance=0.65,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"fontsize": 11},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
            at.set_fontsize(11)

        name_clean = r["display_name"].replace(" ", "_").replace("(", "").replace(")", "").lower()
        ax.set_title(f"Runtime Share — {r['display_name']}\n({r['backend']})", fontsize=13, pad=12, fontweight="bold")
        fig.text(0.5, 0.03, f"Total Latency: {tot:,.1f} ms | 9 Volume Pairs", ha="center", fontsize=10, color="#555555")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        save_fig(fig, f"runtime_share_{name_clean}")

def main():
    records = load_data()
    
    # Save combined summary json
    summary_path = OUTPUT_DIR / "deployment_breakdown_summary.json"
    summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {summary_path.name}")

    # 1. Bar charts
    plot_bar_chart(records, "model_ms", "model_ms_sd", "Model Execution Latency — FP32 vs. INT8", "Milliseconds (log scale)", "model_execution_latency", log_scale=True, format_str="{:,.1f} ms")
    plot_bar_chart(records, "data_ms", None, "Data Movement + Warping Latency — FP32 vs. INT8", "Milliseconds (log scale)", "data_movement_warp_latency", log_scale=True, format_str="{:,.1f} ms")
    plot_bar_chart(records, "total_ms", "total_ms_sd", "Total Inference-to-Warp Latency — FP32 vs. INT8", "Milliseconds (log scale)", "total_inference_warp_latency", log_scale=True, format_str="{:,.1f} ms")
    plot_bar_chart(records, "power_w", "power_w_sd", "Mean Power Draw During Execution — FP32 vs. INT8", "Watts (W)", "mean_power_draw", log_scale=False, format_str="{:.2f} W")
    plot_bar_chart(records, "energy_j", "energy_j_sd", "Raw Energy per Inference — FP32 vs. INT8", "Joules (J)", "raw_energy_per_inference", log_scale=False, format_str="{:.2f} J")
    plot_bar_chart(records, "perf_per_w", None, "Energy Efficiency — FP32 vs. INT8", "Inferences / Second / Watt", "energy_efficiency_perf_per_watt", log_scale=False, format_str="{:.4f}")

    # 2. Stacked chart
    plot_stacked_chart(records, "stacked_runtime_breakdown")

    # 3. Pie charts for each platform
    plot_pie_charts(records)

    print("All deployment breakdown charts generated successfully in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
