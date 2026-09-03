#!/usr/bin/env python3
"""Create the consolidated V4 precision, power, and efficiency comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "Voxelmorph" / "artifacts" / "results"
BOARD_PATH = (
    RESULTS_ROOT
    / "inference_pipeline_breakdown"
    / "inference_pipeline_power_latency_v4.json"
)
HOST_PATH = (
    RESULTS_ROOT
    / "inference_only_local_power_latency"
    / "v4_fp32_int8_host_comparison.json"
)
GPU_INT8_PATH = (
    RESULTS_ROOT
    / "inference_only_local_power_latency"
    / "v4_gpu_int8_tensorrt.json"
)
ARM_INT8_PATH = RESULTS_ROOT / "int8_arm_board_results_v4.json"
ARM_INT8_BUNDLE_PATH = (
    REPO_ROOT / "fpga_inference_int8_board_bundle" / "int8_arm_board_results_v4.json"
)
OUTPUT_DIR = RESULTS_ROOT / "presentation_v4_precision_comparison"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_record(
    platform: str,
    precision: str,
    model: dict,
    latency_key: str,
    backend: str,
    power_scope: str,
) -> dict:
    latency_ms = float(model[latency_key])
    power_w = float(model["power_mean_w_mean"])
    throughput = 1000.0 / latency_ms
    return {
        "platform": platform,
        "precision": precision,
        "status": "measured",
        "pair_count": int(model["pair_count"]),
        "latency_ms_mean": latency_ms,
        "latency_ms_sd": float(
            model[latency_key.replace("_mean", "_sd")]
        ),
        "raw_mean_power_w": power_w,
        "raw_mean_power_sd_w": float(model["power_mean_w_sd"]),
        "raw_energy_j_per_inference": float(
            model["energy_j_per_inference_mean"]
        ),
        "raw_energy_j_per_inference_sd": float(
            model["energy_j_per_inference_sd"]
        ),
        "throughput_inferences_per_s": throughput,
        "performance_per_watt_inferences_per_s_per_w": throughput / power_w,
        "backend": backend,
        "power_scope": power_scope,
    }


def build_records() -> list[dict]:
    board = read_json(BOARD_PATH)["models"]
    host = read_json(HOST_PATH)["models"]
    gpu_int8 = read_json(GPU_INT8_PATH)["models"]["gpu_int8"]
    board_scope = "monitored PSINTFP + INT rails"
    desktop_scope = "CPU PPT/package + NVIDIA GPU board"

    arm_int8_file = None
    if ARM_INT8_PATH.exists():
        arm_int8_file = ARM_INT8_PATH
    elif ARM_INT8_BUNDLE_PATH.exists():
        arm_int8_file = ARM_INT8_BUNDLE_PATH

    if arm_int8_file is not None:
        arm_int8_model = read_json(arm_int8_file)["models"]["2.5d_fused_arm_cpu"]
        arm_int8_record = make_record(
            "ARM CPU",
            "INT8",
            arm_int8_model,
            "total_runtime_ms_mean",
            "PyTorch INT8 Quantized",
            board_scope,
        )
    else:
        arm_int8_record = {
            "platform": "ARM CPU",
            "precision": "INT8",
            "status": "pending_board_connection",
            "reason": (
                "ZCU104 is offline; run measure_v4_arm_int8.py from the "
                "board V4 notebook when connected."
            ),
        }

    records = [
        make_record(
            "FPGA DPU",
            "INT8",
            board["2.5d_fused_dpu"],
            "total_runtime_ms_mean",
            "Vitis AI 2.5 DPU",
            board_scope,
        ),
        make_record(
            "ARM CPU",
            "FP32",
            board["2.5d_fused_arm_cpu"],
            "total_runtime_ms_mean",
            "PyTorch FP32",
            board_scope,
        ),
        arm_int8_record,
        make_record(
            "Local CPU",
            "FP32",
            host["cpu_fp32"],
            "latency_ms_mean",
            "PyTorch FP32",
            desktop_scope,
        ),
        make_record(
            "Local CPU",
            "INT8",
            host["cpu_int8"],
            "latency_ms_mean",
            "ONNX Runtime QOperator INT8",
            desktop_scope,
        ),
        make_record(
            "Local GPU",
            "FP32",
            host["gpu_fp32"],
            "latency_ms_mean",
            "PyTorch FP32 CUDA",
            desktop_scope,
        ),
        make_record(
            "Local GPU",
            "INT8",
            gpu_int8,
            "latency_ms_mean",
            "TensorRT INT8 from Vitis Q/DQ ONNX",
            desktop_scope,
        ),
    ]
    return records


def grouped_values(records: list[dict], field: str):
    platforms = ["FPGA DPU", "ARM CPU", "Local CPU", "Local GPU"]
    precisions = ["FP32", "INT8"]
    values = np.full((len(precisions), len(platforms)), np.nan)
    errors = np.full_like(values, np.nan)
    error_field = {
        "latency_s": "latency_ms_sd",
        "raw_mean_power_w": "raw_mean_power_sd_w",
        "raw_energy_j_per_inference": "raw_energy_j_per_inference_sd",
    }.get(field)
    for record in records:
        if record.get("status") != "measured":
            continue
        i = precisions.index(record["precision"])
        j = platforms.index(record["platform"])
        if field == "latency_s":
            values[i, j] = record["latency_ms_mean"] / 1000.0
            if error_field:
                errors[i, j] = record[error_field] / 1000.0
        else:
            values[i, j] = record[field]
            if error_field:
                errors[i, j] = record[error_field]
    return platforms, precisions, values, errors


def label_bars(axis, containers, fmt: str) -> None:
    for container in containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append("" if not np.isfinite(height) else format(height, fmt))
        axis.bar_label(container, labels=labels, padding=3, fontsize=8)


def plot_panel(axis, records, field, title, ylabel, log_scale=False):
    platforms, precisions, values, errors = grouped_values(records, field)
    x = np.arange(len(platforms))
    width = 0.34
    colors = {"FP32": "#3977B8", "INT8": "#E57A2D"}
    containers = []
    for index, precision in enumerate(precisions):
        positions = x + (index - 0.5) * width
        yerr = errors[index] if np.isfinite(errors[index]).any() else None
        container = axis.bar(
            positions,
            values[index],
            width,
            label=precision,
            color=colors[precision],
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
        axis.set_ylim(ymin, ymax * 6.0)
    else:
        ymin, ymax = axis.get_ylim()
        axis.set_ylim(ymin, ymax * 1.3)
    label_bars(axis, containers, ".3g")
    if field in {"latency_s", "raw_mean_power_w"}:
        arm_int8_rec = [r for r in records if r["platform"] == "ARM CPU" and r["precision"] == "INT8"]
        if arm_int8_rec and arm_int8_rec[0].get("status") != "measured":
            axis.text(
                x[1] + width / 2,
                axis.get_ylim()[0] * (1.7 if log_scale else 1.05),
                "pending",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#777777",
                rotation=90,
            )
    return containers


def save_single(records, field, title, ylabel, filename, log_scale=False, legend_loc="upper right"):
    figure, axis = plt.subplots(figsize=(11, 5.8))
    containers = plot_panel(
        axis, records, field, title, ylabel, log_scale=log_scale
    )
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
    figure.savefig(OUTPUT_DIR / filename, format="svg", bbox_inches="tight")
    png_filename = Path(filename).stem + ".png"
    figure.savefig(OUTPUT_DIR / png_filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    arm_int8_measured = any(
        r["platform"] == "ARM CPU" and r["precision"] == "INT8" and r.get("status") == "measured"
        for r in records
    )
    complete_status = [
        "FPGA DPU INT8",
        "ARM CPU FP32",
        "Local CPU FP32",
        "Local CPU INT8",
        "Local GPU FP32",
        "Local GPU INT8",
    ]
    if arm_int8_measured:
        complete_status.append("ARM CPU INT8")
        pending_status = []
    else:
        pending_status = ["ARM CPU INT8"]

    payload = {
        "schema_version": 1,
        "measurement_status": {
            "complete": complete_status,
            "pending": pending_status,
        },
        "comparison_warning": (
            "Board and desktop power values cover different physical scopes; "
            "neither is whole-system wall power."
        ),
        "records": records,
    }
    (OUTPUT_DIR / "v4_precision_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    panels = [
        (
            "latency_s",
            "Inference-to-warp latency",
            "Seconds (log scale)",
            True,
        ),
        (
            "raw_mean_power_w",
            "Raw mean active power",
            "Watts (log scale)",
            True,
        ),
        (
            "raw_energy_j_per_inference",
            "Raw energy per inference",
            "Joules (log scale)",
            True,
        ),
        (
            "performance_per_watt_inferences_per_s_per_w",
            "Performance per watt",
            "Inferences/s/W (log scale)",
            True,
        ),
    ]
    containers = None
    for axis, panel in zip(axes.flat, panels):
        containers = plot_panel(axis, records, *panel)
    figure.legend(
        handles=[containers[0], containers[1]],
        labels=["FP32", "INT8"],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
    )
    figure.suptitle(
        "2.5D deployment: precision and hardware",
        x=0.05,
        y=0.992,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.012,
        (
            "9 paired volumes • mean ± SD • raw power is not idle-subtracted • "
            "board rails and desktop sensors are different measurement scopes"
        ),
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    figure.savefig(
        OUTPUT_DIR / "v4_precision_deployment_comparison.svg",
        format="svg",
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT_DIR / "v4_precision_deployment_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    save_single(
        records,
        "latency_s",
        "Inference-to-warp latency",
        "Seconds (log scale)",
        "v4_precision_latency.svg",
        log_scale=True,
        legend_loc="upper right",
    )
    save_single(
        records,
        "raw_mean_power_w",
        "Raw mean active power",
        "Watts (log scale)",
        "v4_precision_raw_power.svg",
        log_scale=True,
        legend_loc="upper left",
    )
    save_single(
        records,
        "raw_energy_j_per_inference",
        "Raw energy per inference",
        "Joules (log scale)",
        "v4_precision_raw_energy.svg",
        log_scale=True,
        legend_loc="upper center",
    )
    save_single(
        records,
        "performance_per_watt_inferences_per_s_per_w",
        "Performance per watt",
        "Inferences/s/W (log scale)",
        "v4_precision_performance_per_watt.svg",
        log_scale=True,
        legend_loc="upper center",
    )
    print(f"Wrote {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
