#!/usr/bin/env python3
"""Generate an all-model accuracy/efficiency bubble chart.

The chart uses the fixed PubTabNet-5000 validation split. Horizontal position
encodes structure accuracy, vertical position encodes inference speed (FPS),
and bubble area encodes trainable parameters.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "论文材料" / "all_models_lightweight_bubble_20260801"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


# Values are from DBM_SLANet_Tables2_3_Expanded_Final_20260710.md.
# one-way parameters are an estimate from the bidirectional=false module graph.
ROWS = [
    {"model": "SLANet", "short": "SLANet", "group": "Baseline", "accuracy": 68.98, "fps": 23.50, "params": 1.89, "estimated": False},
    {"model": "Cell Token only", "short": "Cell Token", "group": "Baseline", "accuracy": 69.62, "fps": 24.25, "params": 1.93, "estimated": False},
    {"model": "Lite Mamba", "short": "Lite Mamba", "group": "Lightweight", "accuracy": 70.84, "fps": 15.99, "params": 2.09, "estimated": False},
    {"model": "Full Mamba, lambda_pres=0", "short": "Mamba lambda=0", "group": "Ablation", "accuracy": 69.38, "fps": 12.06, "params": 2.45, "estimated": False},
    {"model": "Full Mamba, bidirectional=false", "short": "Mamba one-way", "group": "Ablation", "accuracy": 69.44, "fps": 16.19, "params": 2.28, "estimated": True},
    {"model": "Full Mamba, geometry zero", "short": "Mamba geom=0", "group": "Ablation", "accuracy": 69.36, "fps": 12.32, "params": 2.45, "estimated": False},
    {"model": "Full Mamba, lambda_pres=0.2", "short": "Mamba lambda=0.2", "group": "Ablation", "accuracy": 70.16, "fps": 12.16, "params": 2.45, "estimated": False},
    {"model": "Full Mamba stage4", "short": "Mamba stage4", "group": "Continuation", "accuracy": 71.78, "fps": 12.20, "params": 2.45, "estimated": False},
    {"model": "Full Mamba stage6", "short": "Mamba stage6", "group": "Continuation", "accuracy": 71.78, "fps": 12.13, "params": 2.45, "estimated": False},
    {"model": "TEDS-select continuation", "short": "TEDS-select", "group": "Continuation", "accuracy": 71.44, "fps": 12.22, "params": 2.45, "estimated": False},
    {"model": "Span-aware continuation", "short": "Span-aware", "group": "Continuation", "accuracy": 71.92, "fps": 12.32, "params": 2.45, "estimated": False},
    {"model": "SLANet + Horizontal Mamba", "short": "Horizontal", "group": "Single-axis", "accuracy": 71.94, "fps": 11.68, "params": 2.45, "estimated": False},
    {"model": "SLANet + Vertical Mamba", "short": "Vertical", "group": "Single-axis", "accuracy": 71.88, "fps": 12.00, "params": 2.45, "estimated": False},
    {"model": "DBM-SLANet", "short": "DBM-SLANet (ours)", "group": "Proposed", "accuracy": 71.82, "fps": 9.65, "params": 2.45, "estimated": False},
]

COLORS = {
    "Baseline": "#A9B5C1",
    "Lightweight": "#E6B55A",
    "Ablation": "#AF9AD0",
    "Continuation": "#7FA7CB",
    "Single-axis": "#59A3A6",
    "Proposed": "#173F4F",
}


def write_source_csv() -> None:
    with (OUT / "all_models_lightweight_bubble_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "short_label", "group", "structure_accuracy_percent", "fps", "params_m", "params_status"])
        for row in ROWS:
            writer.writerow(
                [
                    row["model"], row["short"], row["group"], row["accuracy"], row["fps"], row["params"],
                    "estimated_from_graph" if row["estimated"] else "measured_or_recorded",
                ]
            )


def _bubble_area(params: float) -> float:
    # Area, not radius, represents parameter count. The range stays readable
    # beside the tightly clustered Full-Mamba points.
    return 250.0 + 390.0 * (params / 2.45) ** 2


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(8.2, 5.4), facecolor="white")
    ax = fig.add_axes([0.10, 0.14, 0.87, 0.79])
    ax.set_facecolor("#FFFFFF")

    for group, color in COLORS.items():
        subset = [r for r in ROWS if r["group"] == group]
        ax.scatter(
            [r["accuracy"] for r in subset],
            [r["fps"] for r in subset],
            s=[_bubble_area(r["params"]) for r in subset],
            c=color,
            alpha=0.84 if group != "Proposed" else 0.96,
            edgecolors="#FFFFFF",
            linewidths=1.2,
            zorder=3,
            label=group,
        )

    # A heavier outline makes the proposed model easy to locate without
    # implying that it is the absolute accuracy leader on this split.
    ours = next(r for r in ROWS if r["model"] == "DBM-SLANet")
    ax.scatter([ours["accuracy"]], [ours["fps"]], s=_bubble_area(ours["params"]) + 110,
               facecolors="none", edgecolors="#082A36", linewidths=2.0, zorder=4)

    # Direct labels replace the separate model index. Offsets are in display
    # points so the dense high-accuracy cluster remains readable.
    offsets = {
        "SLANet": (-11, -17),
        "Cell Token only": (8, 17),
        "Lite Mamba": (16, 3),
        "Full Mamba, lambda_pres=0": (-30, -18),
        "Full Mamba, bidirectional=false": (-16, 19),
        "Full Mamba, geometry zero": (-30, 18),
        "Full Mamba, lambda_pres=0.2": (-24, 19),
        "Full Mamba stage4": (-42, 28),
        "Full Mamba stage6": (-42, 5),
        "TEDS-select continuation": (-35, -20),
        "Span-aware continuation": (24, 29),
        "SLANet + Horizontal Mamba": (27, -17),
        "SLANet + Vertical Mamba": (27, 5),
        "DBM-SLANet": (23, -14),
    }
    display_names = {
        "SLANet": "SLANet",
        "Cell Token only": "Cell Token",
        "Lite Mamba": "Lite Mamba",
        "Full Mamba, lambda_pres=0": "Mamba lambda=0",
        "Full Mamba, bidirectional=false": "Mamba one-way",
        "Full Mamba, geometry zero": "Mamba geom=0",
        "Full Mamba, lambda_pres=0.2": "Mamba lambda=0.2",
        "Full Mamba stage4": "Mamba stage4",
        "Full Mamba stage6": "Mamba stage6",
        "TEDS-select continuation": "TEDS-select",
        "Span-aware continuation": "Span-aware",
        "SLANet + Horizontal Mamba": "Horizontal",
        "SLANet + Vertical Mamba": "Vertical",
        "DBM-SLANet": "DBM-SLANet (ours)",
    }
    for row in ROWS:
        dx, dy = offsets[row["model"]]
        ax.annotate(
            display_names[row["model"]],
            xy=(row["accuracy"], row["fps"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.2,
            color="#12313D" if row["model"] == "DBM-SLANet" else "#25333F",
            fontweight="bold" if row["model"] == "DBM-SLANet" else "normal",
            ha="left" if dx >= 0 else "right",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#AAB5BE", "lw": 0.65,
                        "shrinkA": 3, "shrinkB": 3, "connectionstyle": "arc3,rad=0"},
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.84},
            zorder=5,
        )

    ax.set_xlim(68.55, 72.65)
    ax.set_ylim(8.1, 26.0)
    ax.set_xticks([69, 70, 71, 72])
    ax.set_yticks([10, 15, 20, 25])
    ax.grid(True, axis="both", color="#D9DEE3", linewidth=0.75, alpha=0.78)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#35424E")
    ax.spines["bottom"].set_color("#35424E")
    ax.set_xlabel("Structure Accuracy (%)", labelpad=8, color="#24313D")
    ax.set_ylabel("Inference speed (FPS; higher = faster)", labelpad=8, color="#24313D")
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_source_csv()
    fig = build_figure()
    stem = OUT / "all_models_lightweight_bubble"
    fig.savefig(f"{stem}.png", dpi=800, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
