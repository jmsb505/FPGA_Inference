"""Refresh deployment charts with the direct FPGA inference-only measurements."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VOX_DIR = Path(__file__).resolve().parent
GENERATOR = VOX_DIR / "prepare_presentation_comparison_graphs.py"
namespace = runpy.run_path(str(GENERATOR))
OUTPUT_DIR = namespace["OUTPUT_DIR"]
LOCAL_TIMING = VOX_DIR / "artifacts" / "results" / "inference_only_local_power_latency" / "local_2p5d_3d_cpu_gpu_9_pairs.json"
POWER_LATENCY = VOX_DIR / "artifacts" / "results" / "inference_pipeline_breakdown" / "inference_pipeline_power_latency_v4.json"

CONFIG_ORDER = ("Local CPU", "Local CUDA GPU", "FPGA ARM CPU", "FPGA DPU")
CONFIG_COLORS = {
    "FPGA DPU": "#1565C0",
    "FPGA ARM CPU": "#6A1B9A",
    "Local CPU": "#EF6C00",
    "Local CUDA GPU": "#2E7D32",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _deployment_summary():
    local = _load_json(LOCAL_TIMING)
    board = _load_json(POWER_LATENCY)

    def local_row(model_key):
        model = local["models"][model_key]
        total = float(model["latency_ms_mean"])
        model_ms = float(model["model_inference_ms_mean"])
        return {
            "model_inference_ms": model_ms,
            "model_inference_sd": float(model["model_inference_ms_sd"]),
            "data_movement_warp_ms": total - model_ms,
            "total_runtime_ms": total,
            "total_runtime_sd": float(model["latency_ms_sd"]),
            "power_mean_w": float(model["power_mean_w_mean"]),
            "power_mean_w_sd": float(model["power_mean_w_sd"]),
            "energy_j": float(model["energy_j_per_inference_mean"]),
            "energy_j_sd": float(model["energy_j_per_inference_sd"]),
            "dynamic_energy_j": float(model["dynamic_energy_j_per_inference_mean"]),
            "dynamic_energy_j_sd": float(model["dynamic_energy_j_per_inference_sd"]),
        }

    def board_row(model_key):
        model = board["models"][model_key]
        total = float(model["total_runtime_ms_mean"])
        model_ms = float(model["model_inference_ms_mean"])
        return {
            "model_inference_ms": model_ms,
            "model_inference_sd": float(model["model_inference_ms_sd"]),
            "data_movement_warp_ms": total - model_ms,
            "total_runtime_ms": total,
            "total_runtime_sd": float(model["total_runtime_ms_sd"]),
            "power_mean_w": float(model["power_mean_w_mean"]),
            "power_mean_w_sd": float(model["power_mean_w_sd"]),
            "energy_j": float(model["energy_j_per_inference_mean"]),
            "energy_j_sd": float(model["energy_j_per_inference_sd"]),
            "dynamic_energy_j": float(model["dynamic_energy_j_per_inference_mean"]),
            "dynamic_energy_j_sd": float(model["dynamic_energy_j_per_inference_sd"]),
        }

    return {
        "Local CPU": local_row("2.5d_cpu"),
        "Local CUDA GPU": local_row("2.5d_gpu"),
        "FPGA ARM CPU": board_row("2.5d_fused_arm_cpu"),
        "FPGA DPU": board_row("2.5d_fused_dpu"),
    }


def _save_figure(fig, stem):
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path.name, svg_path.name


def _bar_chart(labels, values, colors, title, ylabel, stem, *, errors=None, format_value=None, log_scale=False, note=""):
    fig, axis = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    errors = np.zeros(len(values)) if errors is None else np.asarray(errors, dtype=float)
    bars = axis.bar(x, values, yerr=errors, capsize=5, color=colors, edgecolor="#333333", linewidth=0.8, width=0.62, zorder=3)
    axis.set_title(title, fontsize=15, pad=12)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    axis.set_axisbelow(True)
    if log_scale:
        axis.set_yscale("log")
    for bar, value, error in zip(bars, values, errors):
        text = format_value(value) if format_value else f"{value:.2f}"
        offset = max(error, abs(value) * 0.012, 1e-6)
        axis.text(bar.get_x() + bar.get_width() / 2, value + offset, text, ha="center", va="bottom", fontsize=10)
    if note:
        fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.055 if note else 0.02, 1, 1))
    return _save_figure(fig, stem)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _deployment_summary()
    labels = list(CONFIG_ORDER)
    colors = [CONFIG_COLORS[label] for label in labels]
    generated = {"runtime_bar_files": [], "energy_bar_files": [], "runtime_pie_files": []}

    runtime_specs = [
        ("runtime_model_ms", "Model execution latency", "Milliseconds (log scale)", "model_inference_ms", "model_inference_sd"),
        ("runtime_postprocess_ms", "Data movement + warping latency", "Milliseconds (log scale)", "data_movement_warp_ms", None),
        ("runtime_total_ms", "Inference-to-warp latency", "Milliseconds (log scale)", "total_runtime_ms", "total_runtime_sd"),
    ]
    for stem, title, ylabel, value_key, error_key in runtime_specs:
        values = [rows[label][value_key] for label in labels]
        errors = [rows[label][error_key] if error_key else 0.0 for label in labels]
        generated["runtime_bar_files"].extend(_bar_chart(
            labels,
            values,
            colors,
            f"{title} — 2.5D mean fusion",
            ylabel,
            f"fpga_vs_local_2p5d_mean_fusion_{stem}",
            errors=errors,
            format_value=lambda value: f"{value:,.1f}",
            log_scale=True,
            note="Matched inference-to-warp boundary across 9 aligned volume pairs; quality metrics and disk I/O excluded.",
        ))

    energy_specs = [
        ("energy_absolute_j", "Absolute measured energy", "Joules per inference-to-warp execution", "energy_j", "energy_j_sd"),
        ("energy_dynamic_j", "Idle-baseline-subtracted energy", "Dynamic joules per inference-to-warp execution", "dynamic_energy_j", "dynamic_energy_j_sd"),
        ("mean_power_w", "Mean measured power during execution", "Mean measured power (W)", "power_mean_w", "power_mean_w_sd"),
    ]
    for stem, title, ylabel, value_key, error_key in energy_specs:
        generated["energy_bar_files"].extend(_bar_chart(
            labels,
            [rows[label][value_key] for label in labels],
            colors,
            f"{title} — 2.5D mean fusion",
            ylabel,
            f"fpga_vs_local_2p5d_mean_fusion_{stem}",
            errors=[rows[label][error_key] for label in labels],
            format_value=lambda value: f"{value:.2f}",
            note="Matched inference-to-warp boundary across 9 pairs; board and desktop sensor rails are platform-specific.",
        ))

    for label in labels:
        row = rows[label]
        values = [row["model_inference_ms"], row["data_movement_warp_ms"]]
        fig, axis = plt.subplots(figsize=(6.6, 5.8))
        _, _, autotexts = axis.pie(
            values,
            labels=["Model execution", "Data movement + warping"],
            colors=["#3B73A8", "#BDBDBD"],
            startangle=90,
            counterclock=False,
            autopct=lambda pct: f"{pct:.1f}%",
            pctdistance=0.70,
            wedgeprops={"edgecolor": "white", "linewidth": 1.4},
            textprops={"fontsize": 10},
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        axis.set_title(f"Inference-to-warp runtime share — {label}", fontsize=15, pad=14)
        fig.text(0.5, 0.035, f"Total: {row['total_runtime_ms']:,.1f} ms | Mean fusion | 9 aligned volume pairs", ha="center", fontsize=10, color="#555555")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        safe_label = label.lower().replace(" ", "_").replace("-", "")
        generated["runtime_pie_files"].extend(_save_figure(fig, f"runtime_share_{safe_label}"))

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update({
        "deployment_scope": "Matched inference-to-warp boundary: normalized in-memory volumes through warped intensity output; quality metrics and disk I/O excluded.",
        "runtime_bar_files": generated["runtime_bar_files"],
        "energy_bar_files": generated["energy_bar_files"],
        "runtime_pie_files": generated["runtime_pie_files"],
        "deployment_summary": rows,
        "direct_fpga_source": str(POWER_LATENCY),
        "local_inference_source": str(LOCAL_TIMING),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **generated, "direct_fpga_source": str(POWER_LATENCY)}, indent=2))


if __name__ == "__main__":
    main()
