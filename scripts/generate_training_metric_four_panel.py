#!/usr/bin/env python3
"""Generate a real-data four-panel training and evaluation metric figure."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "论文材料" / "training_metric_four_panel_20260720"

LOGS = {
    "DBM-SLANet (Ours)": ROOT
    / "remote_results/phase11_extra_logs_20260710/phase11c_lambda02_train.log",
    "Bidir=false": ROOT
    / "remote_results/phase11_extra_logs_20260710/phase11e_bidirfalse_train.log",
    "Geom zero": ROOT
    / "remote_results/phase11_extra_logs_20260710/phase11f_geomzero_train.log",
}

METRIC_CSV = (
    ROOT / "remote_results/extra_metrics_20260710/extra_structure_metrics_summary.csv"
)

COLORS = {
    "DBM-SLANet (Ours)": "#3F8A63",
    "Bidir=false": "#D47A24",
    "Geom zero": "#3D6FA8",
}

PR_COLORS = {
    "SLANet": "#7B8794",
    "Cell Token only": "#B56A8A",
    "Horizontal Mamba": "#3D6FA8",
    "Vertical Mamba": "#D47A24",
    "DBM-SLANet": "#3F8A63",
}

TRAIN_PATTERN = re.compile(
    r"epoch:\s*\[(\d+)/(\d+)\].*?"
    r"global_step:\s*(\d+).*?"
    r"acc:\s*([0-9.]+).*?"
    r"loss:\s*([0-9.]+).*?"
    r"structure_loss:\s*([0-9.]+).*?"
    r"loc_loss:\s*([0-9.]+).*?"
    r"cell_presence_loss:\s*([0-9.]+)"
)


def parse_train_log(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        match = TRAIN_PATTERN.search(line)
        if match:
            rows.append(tuple(float(value) for value in match.groups()))
    if not rows:
        raise ValueError(f"No training metrics found in {path}")
    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 2])
    return data[order]


def smooth(values: np.ndarray, window: int = 17) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def load_span_pr() -> list[dict[str, str]]:
    selected = set(PR_COLORS)
    rows = []
    with METRIC_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == "CTA-500" and row["method"] in selected:
                rows.append(row)
    if len(rows) != len(selected):
        found = {row["method"] for row in rows}
        raise ValueError(f"Missing CTA-500 precision-recall rows: {selected - found}")
    return rows


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D7DCE2", linewidth=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)


def draw_panels(fig: plt.Figure, axes: np.ndarray) -> None:
    logs = {name: parse_train_log(path) for name, path in LOGS.items()}
    ours = logs["DBM-SLANet (Ours)"]

    ax = axes[0, 0]
    components = [
        ("Total", ours[:, 4], "#263442"),
        ("Structure", ours[:, 5], "#3D6FA8"),
        ("Localization", ours[:, 6], "#D47A24"),
        ("Cell presence", ours[:, 7], "#3F8A63"),
    ]
    for label, values, color in components:
        ax.plot(ours[:, 2], smooth(values), label=label, color=color, linewidth=1.35)
    ax.set_yscale("log")
    ax.set_title("(a) Training Loss", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Global step", fontsize=9)
    ax.set_ylabel("Loss (log scale)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.8, ncol=2, loc="upper right")
    style_axis(ax)

    ax = axes[0, 1]
    for name, data in logs.items():
        ax.plot(
            data[:, 2],
            smooth(data[:, 4]),
            label=name,
            color=COLORS[name],
            linewidth=1.45,
        )
    ax.set_title("(b) Loss vs. Training Step", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Global step", fontsize=9)
    ax.set_ylabel("Total loss", fontsize=9)
    ax.set_xlim(0, 7600)
    ax.legend(frameon=False, fontsize=7.8)
    style_axis(ax)

    ax = axes[1, 0]
    for name, data in logs.items():
        ax.plot(
            data[:, 2],
            smooth(data[:, 3], window=25),
            label=name,
            color=COLORS[name],
            linewidth=1.45,
        )
    ax.set_title("(c) Accuracy vs. Training Step", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Global step", fontsize=9)
    ax.set_ylabel("Training accuracy", fontsize=9)
    ax.set_xlim(0, 7600)
    ax.set_ylim(0.60, 0.76)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    style_axis(ax)

    ax = axes[1, 1]
    for row in load_span_pr():
        method = row["method"]
        precision = float(row["span_precision"])
        recall = float(row["span_recall"])
        ours_point = method == "DBM-SLANet"
        ax.scatter(
            recall,
            precision,
            s=90 if ours_point else 58,
            color=PR_COLORS[method],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
            label="DBM-SLANet (Ours)" if ours_point else method,
        )
    ax.set_title("(d) Precision-Recall", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Span recall", fontsize=9)
    ax.set_ylabel("Span precision", fontsize=9)
    ax.set_xlim(0.49, 0.63)
    ax.set_ylim(0.67, 0.73)
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    style_axis(ax)


def save_individual_panels(combined_fig: plt.Figure, axes: np.ndarray) -> None:
    combined_fig.canvas.draw()
    renderer = combined_fig.canvas.get_renderer()
    for label, ax in zip("abcd", axes.ravel()):
        bbox = ax.get_tightbbox(renderer).expanded(1.04, 1.08)
        bbox_inches = bbox.transformed(combined_fig.dpi_scale_trans.inverted())
        combined_fig.savefig(
            OUTPUT_DIR / f"panel_{label}.png",
            dpi=500,
            bbox_inches=bbox_inches,
            facecolor="white",
        )
        combined_fig.savefig(
            OUTPUT_DIR / f"panel_{label}.pdf",
            bbox_inches=bbox_inches,
            facecolor="white",
        )


def save_training_curve_pair() -> None:
    """Export replacement panels (c) and (d) with truthful step-based axes."""
    logs = {name: parse_train_log(path) for name, path in LOGS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.55), facecolor="white")

    ax = axes[0]
    for name, data in logs.items():
        ax.plot(
            data[:, 2],
            smooth(data[:, 4]),
            label=name,
            color=COLORS[name],
            linewidth=1.5,
        )
    ax.set_title("(c) Loss vs. Training Step", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Global step", fontsize=9)
    ax.set_ylabel("Total loss", fontsize=9)
    ax.set_xlim(0, 7600)
    ax.legend(frameon=False, fontsize=7.7, loc="lower left")
    style_axis(ax)

    ax = axes[1]
    for name, data in logs.items():
        ax.plot(
            data[:, 2],
            smooth(data[:, 3], window=25),
            label=name,
            color=COLORS[name],
            linewidth=1.5,
        )
    ax.set_title("(d) Accuracy vs. Training Step", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Global step", fontsize=9)
    ax.set_ylabel("Training accuracy", fontsize=9)
    ax.set_xlim(0, 7600)
    ax.set_ylim(0.60, 0.76)
    ax.legend(frameon=False, fontsize=7.7, loc="lower right")
    style_axis(ax)

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.88, wspace=0.22)
    output_base = OUTPUT_DIR / "training_curves_cd_global_step"
    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    fig.savefig(
        output_base.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, panel_ax in zip("cd", axes):
        bbox = panel_ax.get_tightbbox(renderer).expanded(1.04, 1.08)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(
                OUTPUT_DIR / f"panel_{label}_global_step.{suffix}",
                dpi=500 if suffix == "png" else None,
                bbox_inches=bbox_inches,
                facecolor="white",
            )
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#66717E",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), facecolor="white")
    draw_panels(fig, axes)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.975, wspace=0.24, hspace=0.30)
    fig.savefig(
        OUTPUT_DIR / "training_metrics_four_panel.png",
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / "training_metrics_four_panel.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / "training_metrics_four_panel.svg",
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    save_individual_panels(fig, axes)
    plt.close(fig)
    save_training_curve_pair()
    print(f"Generated training metric figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
