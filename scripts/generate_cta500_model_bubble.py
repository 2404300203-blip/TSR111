#!/usr/bin/env python3
"""Generate a CTA-500 accuracy/quality/speed bubble chart from measured results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "论文材料" / "cta500_model_bubble_20260803"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# All values are measured on the same CTA-500 diagnostic subset. DBM-SLANet's
# structural metrics were independently reproduced on 2026-08-03.
ROWS = [
    {"model": "SLANet", "label": "SLANet", "family": "Baseline", "teds": 0.4843354130, "acc": 0.208, "fps": 16.3023},
    {"model": "Cell Token only", "label": "Cell Token", "family": "Lightweight", "teds": 0.5436438417, "acc": 0.284, "fps": 20.3262},
    {"model": "Full Mamba, lambda_pres=0", "label": "Mamba lambda=0", "family": "Ablation", "teds": 0.5392074837, "acc": 0.282, "fps": 9.3970},
    {"model": "Full Mamba, lambda_pres=0.2", "label": "Full Mamba", "family": "Ablation", "teds": 0.5421500000, "acc": 0.302, "fps": 9.5400},
    {"model": "Full Mamba, aux off", "label": "Mamba aux off", "family": "Ablation", "teds": 0.5421500000, "acc": 0.302, "fps": 18.6800},
    {"model": "Full Mamba, bidirectional=false", "label": "Mamba one-way", "family": "Ablation", "teds": 0.5304000000, "acc": 0.272, "fps": 14.1400},
    {"model": "Full Mamba, geometry zero", "label": "Mamba geom=0", "family": "Ablation", "teds": 0.5392100000, "acc": 0.282, "fps": 11.1200},
    {"model": "Horizontal-only Mamba", "label": "Horizontal", "family": "Single-axis", "teds": 0.6276370150, "acc": 0.420, "fps": 8.6467},
    {"model": "Vertical-only Mamba", "label": "Vertical", "family": "Single-axis", "teds": 0.6257612812, "acc": 0.416, "fps": 9.7617},
    {"model": "DBM-SLANet", "label": "DBM-SLANet (ours)", "family": "Proposed", "teds": 0.6406726544, "acc": 0.432, "fps": 8.1429},
]

COLORS = {
    "Baseline": "#70C1E8",
    "Lightweight": "#FFC857",
    "Ablation": "#7FA9E2",
    "Single-axis": "#3EC6A8",
    "Proposed": "#FF6B5E",
}

# Label offsets are in points; short leaders keep dense points legible.
OFFSETS = {
    "SLANet": (-18, 21),
    "Cell Token only": (16, 18),
    "Full Mamba, lambda_pres=0": (-20, -18),
    "Full Mamba, lambda_pres=0.2": (16, -9),
    "Full Mamba, aux off": (16, -15),
    "Full Mamba, bidirectional=false": (-18, 20),
    "Full Mamba, geometry zero": (17, 17),
    "Horizontal-only Mamba": (-18, 21),
    "Vertical-only Mamba": (-19, -19),
    "DBM-SLANet": (-19, -18),
}


def bubble_area(structure_accuracy: float) -> float:
    """Map structure accuracy to marker area."""
    return 115.0 + 1250.0 * structure_accuracy


def write_source_csv() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cta500_model_bubble_source.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "label", "family", "teds", "structure_accuracy", "fps"],
        )
        writer.writeheader()
        for row in ROWS:
            writer.writerow(
                {
                    "model": row["model"],
                    "label": row["label"],
                    "family": row["family"],
                    "teds": row["teds"],
                    "structure_accuracy": row["acc"],
                    "fps": row["fps"],
                }
            )


def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(8.1, 5.75), facecolor="white")
    ax = fig.add_axes([0.105, 0.13, 0.86, 0.82])
    ax.set_facecolor("#FCFEFF")

    # Draw larger bubbles first so nearby points remain visible.
    for row in sorted(ROWS, key=lambda item: item["acc"], reverse=True):
        proposed = row["family"] == "Proposed"
        ax.scatter(
            row["teds"],
            row["fps"],
            s=bubble_area(row["acc"]),
            color=COLORS[row["family"]],
            alpha=0.98 if proposed else 0.90,
            edgecolor="#C63F38" if proposed else "white",
            linewidth=2.0 if proposed else 1.15,
            zorder=5 if proposed else 3,
        )

    for row in ROWS:
        dx, dy = OFFSETS[row["model"]]
        proposed = row["family"] == "Proposed"
        ax.annotate(
            row["label"],
            xy=(row["teds"], row["fps"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=8.0 if proposed else 7.4,
            fontweight="bold" if proposed else "normal",
            color="#C63F38" if proposed else "#26343C",
            arrowprops={
                "arrowstyle": "-",
                "color": "#78919D",
                "lw": 0.65,
                "shrinkA": 2,
                "shrinkB": 9,
                "connectionstyle": "arc3,rad=0",
            },
            zorder=6,
        )

    ax.text(
        0.477,
        21.55,
        "Bubble area = structure exact match",
        ha="left",
        va="center",
        fontsize=7.7,
        color="#5C7886",
    )

    ax.set_xlim(0.472, 0.661)
    ax.set_ylim(6.5, 22.1)
    ax.set_xticks([0.48, 0.52, 0.56, 0.60, 0.64])
    ax.set_yticks([8, 10, 12, 14, 16, 18, 20, 22])
    ax.set_xlabel("TEDS score on CTA-500", labelpad=9, color="#24323A")
    ax.set_ylabel("Inference speed on CTA-500 (FPS; higher = faster)", labelpad=9, color="#24323A")
    ax.grid(True, color="#D8E7EE", linewidth=0.72, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3D4A51")
    ax.spines["bottom"].set_color("#3D4A51")
    ax.tick_params(colors="#3D4A51", length=3)
    return fig


def main() -> None:
    write_source_csv()
    fig = build_figure()
    stem = OUT / "cta500_model_bubble_bright"
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
