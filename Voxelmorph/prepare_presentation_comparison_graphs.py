"""Generate bar-chart and runtime-share pie-chart figures for the presentation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VOX_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = VOX_DIR / "artifacts" / "results" / "presentation_comparisons"
BOARD_RESULTS = VOX_DIR / "benchmark_results_comparison_v4.json"
LOCAL_RESULTS = VOX_DIR / "artifacts" / "results" / "compare_2p5d_v4_fusion_3d_v2_test" / "benchmark_results_comparison_v4_local.json"
LOCAL_ROWS = VOX_DIR / "artifacts" / "results" / "compare_2p5d_v4_fusion_3d_v2_test" / "v4_vs_3d_rows.json"
RESOURCE_METRICS = VOX_DIR / "artifacts" / "results" / "presentation_inputs" / "presentation_resource_metrics_v4.json"
LOCAL_TIMING = VOX_DIR / "artifacts" / "results" / "inference_only_local_power_latency" / "local_2p5d_3d_cpu_gpu_9_pairs.json"
PLATFORM_SUMMARY = VOX_DIR / "artifacts" / "results" / "compare_v4_local_fpga_results" / "v4_platform_method_summary.csv"

CONFIG_ORDER = ("Local CPU", "Local CUDA GPU", "FPGA ARM CPU", "FPGA DPU")
CONFIG_COLORS = {
    "FPGA DPU": "#1565C0",
    "FPGA ARM CPU": "#6A1B9A",
    "Local CPU": "#EF6C00",
    "Local CUDA GPU": "#2E7D32",
}
GPU_ORDER = ("2.5D mean fusion", "3D")
GPU_COLORS = {"2.5D mean fusion": "#3B73A8", "3D": "#54A24B"}

SSIM_BEFORE_BY_PAIR = {
    0: 0.7167415022850037,
    1: 0.7166081666946411,
    2: 0.7381207942962646,
    3: 0.7355596423149109,
    4: 0.7159125208854675,
    5: 0.7613824009895325,
    6: 0.7441083192825317,
    7: 0.7024828791618347,
    8: 0.741512656211853,
}

QUALITY = {
    "dice": {
        "after": "dice_after",
        "before": "dice_before",
        "title": "Registration overlap",
        "ylabel": "Dice (higher is better)",
        "format": lambda value: f"{value:.4f}",
    },
    "mi": {
        "after": "mi_after",
        "before": "mi_before",
        "title": "Image similarity",
        "ylabel": "Mutual information (higher is better)",
        "format": lambda value: f"{value:.4f}",
    },
    "ssim": {
        "after": "ssim_deformed_fixed",
        "before": None,
        "title": "Structural similarity",
        "ylabel": "SSIM (higher is better)",
        "format": lambda value: f"{value:.4f}",
    },
    "tre": {
        "after": "tre_mm",
        "before": "tre_before_mm",
        "title": "Anatomical label-centroid error",
        "ylabel": "TRE (mm; lower is better)",
        "format": lambda value: f"{value:.3f}",
    },
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_sd(values):
    values = [float(value) for value in values]
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _load_quality_rows():
    board = _load_json(BOARD_RESULTS)
    local = _load_json(LOCAL_RESULTS)
    local_rows = _load_json(LOCAL_ROWS)

    cpu_pairs = sorted(local["cpu"]["pairs"], key=lambda row: row["pair_index"])
    gpu_pairs = sorted(local["cuda"]["pairs"], key=lambda row: row["pair_index"])
    arm_pairs = sorted(board["cpu"]["pairs"], key=lambda row: row["pair_index"])
    dpu_pairs = sorted(board["fpga"]["pairs"], key=lambda row: row["pair_index"])
    gpu_3d = sorted(
        (row for row in local_rows if row["device"] == "cuda" and row["method"] == "3d_v2"),
        key=lambda row: row["pair_index"],
    )
    pair_keys = [(row["pair_index"], row["moving_idx"], row["fixed_idx"]) for row in gpu_pairs]
    if len(pair_keys) != 9 or any(len(rows) != 9 for rows in (cpu_pairs, arm_pairs, dpu_pairs, gpu_3d)):
        raise ValueError("Expected nine aligned held-out pairs for every comparison")

    main_rows = {
        "Local CPU": [row["v4_mean_fused"] for row in cpu_pairs],
        "Local CUDA GPU": [row["v4_mean_fused"] for row in gpu_pairs],
        "FPGA ARM CPU": [row["v4_mean_fused"] for row in arm_pairs],
        "FPGA DPU": [row["v4_mean_fused"] for row in dpu_pairs],
    }
    gpu_rows = {
        "2.5D mean fusion": [row["v4_mean_fused"] for row in gpu_pairs],
        "3D": gpu_3d,
    }
    return pair_keys, main_rows, gpu_rows


def _before_values(rows, pair_keys, metric):
    before_key = QUALITY[metric]["before"]
    if metric == "ssim":
        return [SSIM_BEFORE_BY_PAIR[pair_index] for pair_index, _, _ in pair_keys]
    return [float(row[before_key]) for row in rows]


def _save_figure(fig, stem: str):
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path.name, svg_path.name


def _bar_chart(labels, values, colors, title, ylabel, stem, *, errors=None, baseline=None, baseline_label=None, format_value=None, log_scale=False, note=""):
    fig, axis = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    errors = np.zeros(len(values)) if errors is None else np.asarray(errors, dtype=float)
    bars = axis.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        width=0.62,
        zorder=3,
    )
    if baseline is not None:
        axis.axhline(
            baseline,
            color="#222222",
            linewidth=1.8,
            linestyle="--",
            label=baseline_label or "Before registration",
            zorder=2,
        )
    axis.set_title(title, fontsize=15, pad=12)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, labels)
    axis.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    axis.set_axisbelow(True)
    if log_scale:
        axis.set_yscale("log")
    if baseline is not None and not log_scale:
        lows = [baseline] + [value - error for value, error in zip(values, errors)]
        highs = [baseline] + [value + error for value, error in zip(values, errors)]
        low, high = min(lows), max(highs)
        span = max(high - low, abs(high) * 0.04, 1e-6)
        axis.set_ylim(max(0, low - span * 0.16), high + span * 0.20)
    for bar, value, error in zip(bars, values, errors):
        text = format_value(value) if format_value else f"{value:.2f}"
        offset = max(error, abs(value) * 0.012, 1e-6)
        axis.text(bar.get_x() + bar.get_width() / 2, value + offset, text, ha="center", va="bottom", fontsize=10)
    if baseline is not None:
        axis.legend(frameon=False, loc="upper left")
    if note:
        fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.055 if note else 0.02, 1, 1))
    return _save_figure(fig, stem)


def _quality_chart(metric, rows_by_label, labels, colors, pair_keys, stem, subtitle):
    spec = QUALITY[metric]
    means = []
    errors = []
    for label in labels:
        values = [float(row[spec["after"]]) for row in rows_by_label[label]]
        average, deviation = _mean_sd(values)
        means.append(average)
        errors.append(deviation)
    before = _before_values(rows_by_label[labels[0]], pair_keys, metric)
    before_mean = mean(before)
    return _bar_chart(
        labels,
        means,
        [colors[label] for label in labels],
        f"{spec['title']} — {subtitle}",
        spec["ylabel"],
        stem,
        errors=errors,
        baseline=before_mean,
        baseline_label="Before registration mean",
        format_value=spec["format"],
        note="Mean ± sample SD across 9 aligned volume pairs; mean fusion only for 2.5D.",
    )


def _load_platform_summary():
    with PLATFORM_SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row["configuration"]: row
        for row in rows
        if row["method"] == "v4_mean_fused" and row["configuration"] in CONFIG_ORDER
    }
    if set(selected) != set(CONFIG_ORDER):
        raise ValueError("The platform summary is missing one or more mean-fusion configurations")
    return selected


def _runtime_share_charts(platform_rows):
    files = []
    for label in CONFIG_ORDER:
        row = platform_rows[label]
        model_ms = float(row["model_inference_ms"])
        post_ms = float(row["postprocess_ms"])
        total_ms = float(row["total_runtime_ms"])
        values = [model_ms, post_ms]
        fig, axis = plt.subplots(figsize=(6.6, 5.8))
        wedges, _, autotexts = axis.pie(
            values,
            labels=["Model inference", "Post-processing"],
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
        axis.set_title(f"Runtime share — {label}", fontsize=15, pad=14)
        fig.text(0.5, 0.035, f"Total: {total_ms:,.1f} ms | Mean fusion | 9 aligned volume pairs", ha="center", fontsize=10, color="#555555")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        safe_label = label.lower().replace(" ", "_").replace("-", "")
        files.extend(_save_figure(fig, f"runtime_share_{safe_label}"))
    return files


def create_graphs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_keys, main_rows, gpu_rows = _load_quality_rows()
    generated = {"main_quality_files": [], "gpu_quality_files": [], "resource_files": [], "runtime_bar_files": [], "energy_bar_files": [], "runtime_pie_files": []}

    for metric in QUALITY:
        png, svg = _quality_chart(
            metric,
            main_rows,
            CONFIG_ORDER,
            CONFIG_COLORS,
            pair_keys,
            f"fpga_vs_local_2p5d_mean_fusion_{metric}",
            "2.5D mean fusion across local and FPGA platforms",
        )
        generated["main_quality_files"].extend([png, svg])
        png, svg = _quality_chart(
            metric,
            gpu_rows,
            GPU_ORDER,
            GPU_COLORS,
            pair_keys,
            f"gpu_2p5d_mean_fusion_vs_3d_{metric}",
            "GPU comparison",
        )
        generated["gpu_quality_files"].extend([png, svg])

    resource = _load_json(RESOURCE_METRICS)
    timing = _load_json(LOCAL_TIMING)
    resource_specs = [
        ("parameters", [resource["models"]["2p5d_mean_fused"]["parameter_count"], resource["models"]["3d_v2"]["parameter_count"]], [0, 0], "Parameter count", "Trainable parameters", lambda value: f"{int(value):,}", "Mean model definition; 2.5D uses mean fusion at evaluation time."),
        ("cuda_peak_memory", [resource["models"]["2p5d_mean_fused"]["cuda_peak_memory_mb_mean"], resource["models"]["3d_v2"]["cuda_peak_memory_mb_mean"]], [0, 0], "Peak CUDA memory", "Peak CUDA allocation (MiB)", lambda value: f"{value:.1f}", "Mean across 9 GPU pairs."),
        ("inference_to_warp", [timing["models"]["2.5d_gpu"]["latency_ms_mean"], timing["models"]["3d_gpu"]["latency_ms_mean"]], [timing["models"]["2.5d_gpu"]["latency_ms_sd"], timing["models"]["3d_gpu"]["latency_ms_sd"]], "Inference-to-warp time", "Time per volume pair (ms)", lambda value: f"{value:.1f}", "Mean ± sample SD across 9 GPU pairs; quality metrics and disk I/O excluded."),
    ]
    for name, values, errors, title, ylabel, formatter, note in resource_specs:
        ratio = values[1] / values[0] if name != "inference_to_warp" else values[0] / values[1]
        annotation = f"3D uses {ratio:.2f}× more" if name == "parameters" else (f"3D uses {ratio:.1f}× more" if name == "cuda_peak_memory" else f"2.5D takes {ratio:.2f}× longer")
        png, svg = _bar_chart(
            list(GPU_ORDER),
            values,
            [GPU_COLORS[label] for label in GPU_ORDER],
            f"{title} — 2.5D mean fusion vs 3D GPU",
            ylabel,
            f"gpu_2p5d_mean_fusion_vs_3d_{name}",
            errors=errors,
            format_value=formatter,
            note=f"{annotation}. {note}",
        )
        generated["resource_files"].extend([png, svg])

    platform_rows = _load_platform_summary()
    platform_labels = list(CONFIG_ORDER)
    platform_colors = [CONFIG_COLORS[label] for label in platform_labels]
    runtime_specs = [
        ("runtime_model_ms", "Model inference latency", "Milliseconds (log scale)", "model_inference_ms"),
        ("runtime_postprocess_ms", "Post-processing latency", "Milliseconds (log scale)", "postprocess_ms"),
        ("runtime_total_ms", "End-to-end registration latency", "Milliseconds (log scale)", "total_runtime_ms"),
    ]
    for stem, title, ylabel, key in runtime_specs:
        values = [float(platform_rows[label][key]) for label in platform_labels]
        png, svg = _bar_chart(
            platform_labels,
            values,
            platform_colors,
            f"{title} — 2.5D mean fusion",
            ylabel,
            f"fpga_vs_local_2p5d_mean_fusion_{stem}",
            format_value=lambda value: f"{value:,.1f}",
            log_scale=True,
            note="Recorded full-evaluation window across 9 aligned volume pairs; use pies for percentage breakdown.",
        )
        generated["runtime_bar_files"].extend([png, svg])

    energy_specs = [
        ("energy_absolute_j", "Absolute measured energy", "Joules per registration", "energy_j"),
        ("energy_dynamic_j", "Idle-baseline-subtracted energy", "Dynamic joules per registration", "dynamic_energy_j"),
    ]
    for stem, title, ylabel, key in energy_specs:
        values = [float(platform_rows[label][key]) for label in platform_labels]
        png, svg = _bar_chart(
            platform_labels,
            values,
            platform_colors,
            f"{title} — 2.5D mean fusion",
            ylabel,
            f"fpga_vs_local_2p5d_mean_fusion_{stem}",
            format_value=lambda value: f"{value:.2f}",
            note="Platform-specific sensor scopes; do not interpret board and desktop joules as identical whole-system power.",
        )
        generated["energy_bar_files"].extend([png, svg])

    generated["runtime_pie_files"] = _runtime_share_charts(platform_rows)

    manifest = {
        "chart_style": "Grouped bar charts for quality, resource, latency, and energy metrics; one pie chart per device for runtime percentage breakdown.",
        "scope": "Mean fusion only for 2.5D results; 3D is the non-fused GPU reference.",
        "pair_count": 9,
        "quality_metrics": list(QUALITY),
        "main_quality_files": generated["main_quality_files"],
        "gpu_quality_files": generated["gpu_quality_files"],
        "resource_files": generated["resource_files"],
        "runtime_bar_files": generated["runtime_bar_files"],
        "energy_bar_files": generated["energy_bar_files"],
        "runtime_pie_files": generated["runtime_pie_files"],
        "source_files": [str(BOARD_RESULTS), str(LOCAL_RESULTS), str(LOCAL_ROWS), str(RESOURCE_METRICS), str(LOCAL_TIMING), str(PLATFORM_SUMMARY)],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = create_graphs()
    print(json.dumps({key: value for key, value in result.items() if key.endswith("files")}, indent=2))
