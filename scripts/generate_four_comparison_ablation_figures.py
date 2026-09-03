#!/usr/bin/env python3
"""Generate four paper-ready DBM-SLANet comparison and ablation figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#3569A8"
CYAN = "#4B9B9B"
ORANGE = "#D88935"
GREEN = "#4F8A63"
RED = "#B6534F"
PURPLE = "#7762A8"
GRAY = "#A7AFBA"
DARK = "#263442"
GRID = "#D9DEE5"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": DARK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def comparison_general(output_dir: Path) -> None:
    datasets = ["PubTabNet-5000", "CTA500"]
    baseline_teds = np.array([0.8331, 0.4843])
    dbm_teds = np.array([0.8440, 0.6407])
    baseline_acc = np.array([0.6898, 0.2080])
    dbm_acc = np.array([0.7182, 0.4320])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.25))
    x = np.arange(len(datasets))
    width = 0.32
    panels = [
        (axes[0], baseline_teds, dbm_teds, "TEDS", "a"),
        (axes[1], baseline_acc, dbm_acc, "Structure accuracy", "b"),
    ]

    for ax, baseline, dbm, title, label in panels:
        bars_base = ax.bar(
            x - width / 2,
            baseline,
            width,
            color=GRAY,
            edgecolor="white",
            label="SLANet",
        )
        bars_dbm = ax.bar(
            x + width / 2,
            dbm,
            width,
            color=BLUE,
            edgecolor="white",
            label="DBM-SLANet",
        )
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks(x, datasets)
        ax.set_ylim(0, 0.95)
        ax.set_ylabel("Score")
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        add_panel_label(ax, label)

        for bar, value in zip(bars_base, baseline):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 0.045,
                f"{value:.4f}",
                ha="center",
                va="top",
                fontsize=7.5,
                color=DARK,
            )
        for idx, (bar, value) in enumerate(zip(bars_dbm, dbm)):
            gain = value - baseline[idx]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.4f}\n(+{gain:.4f})",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=BLUE,
                fontweight="bold",
            )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
    )
    fig.suptitle(
        "General and complex-table recognition performance",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.2)
    save_figure(fig, output_dir, "comparison_1_pubtabnet_cta500")


def comparison_financial(output_dir: Path) -> None:
    settings = [
        "Pilot-75",
        "Company-\ndisjoint-100",
        "Independent-\n500",
        "Company-\nheldout-100",
    ]
    baseline_teds = np.array([0.4165, 0.4739, 0.4396, 0.4211])
    dbm_teds = np.array([0.5520, 0.6127, 0.4877, 0.4558])
    baseline_acc = np.array([0.1467, 0.2000, 0.0540, 0.0200])
    dbm_acc = np.array([0.2667, 0.3500, 0.0660, 0.0700])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.45))
    x = np.arange(len(settings))
    width = 0.34
    panels = [
        (axes[0], baseline_teds, dbm_teds, "TEDS", 0.72, "a"),
        (axes[1], baseline_acc, dbm_acc, "Structure accuracy", 0.44, "b"),
    ]

    for ax, baseline, dbm, title, ymax, label in panels:
        ax.bar(x - width / 2, baseline, width, color=GRAY, label="SLANet baseline-ft")
        ax.bar(x + width / 2, dbm, width, color=BLUE, label="DBM-R2")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks(x, settings)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Score")
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        add_panel_label(ax, label)
        for idx, value in enumerate(dbm):
            ax.text(
                idx + width / 2,
                value + ymax * 0.035,
                f"+{value - baseline[idx]:.3f}",
                ha="center",
                va="bottom",
                color=BLUE,
                fontsize=7.5,
                fontweight="bold",
            )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
    )
    fig.suptitle(
        "Financial-domain adaptation consistently improves table recognition",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.1)
    save_figure(fig, output_dir, "comparison_2_fintabnet")


def ablation_components(output_dir: Path) -> None:
    methods = [
        "Cell token only",
        "Full model, $\\lambda_{pres}=0$",
        "Full model",
        "No bidirectional scan",
        "No geometry input",
    ]
    teds = np.array([0.54364, 0.53921, 0.54215, 0.53040, 0.53921])
    acc = np.array([0.284, 0.282, 0.302, 0.272, 0.282])
    y = np.arange(len(methods))[::-1]
    colors = [GRAY, ORANGE, BLUE, CYAN, PURPLE]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.55))
    panels = [
        (axes[0], teds, "TEDS", (0.525, 0.550), "a"),
        (axes[1], acc, "Structure accuracy", (0.265, 0.308), "b"),
    ]

    for ax, values, title, xlim, label in panels:
        ax.hlines(y, xlim[0], values, color=GRID, linewidth=2.2)
        ax.scatter(values, y, s=62, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_yticks(y, methods)
        ax.set_xlim(*xlim)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        add_panel_label(ax, label)
        for value, ypos, color in zip(values, y, colors):
            ax.text(
                value + (xlim[1] - xlim[0]) * 0.018,
                ypos,
                f"{value:.3f}",
                va="center",
                fontsize=7.5,
                color=color,
                fontweight="bold" if color == BLUE else "normal",
            )

    axes[1].text(
        0.302,
        y[2] + 0.42,
        "+2.0 pp vs $\\lambda_{pres}=0$",
        color=BLUE,
        fontsize=7.5,
        ha="center",
        fontweight="bold",
    )
    fig.suptitle(
        "Controlled component ablations on CTA500",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout(w_pad=2.4)
    save_figure(fig, output_dir, "ablation_1_components_cta500")


def ablation_tradeoff(output_dir: Path) -> None:
    names = [
        "Cell token",
        "$\\lambda_{pres}=0$",
        "Full model",
        "No bidirectional",
        "No geometry",
        "Aux off",
    ]
    fps = np.array([20.33, 9.40, 9.54, 14.14, 11.12, 18.68])
    teds = np.array([0.54364, 0.53921, 0.54215, 0.53040, 0.53921, 0.54215])
    acc = np.array([0.284, 0.282, 0.302, 0.272, 0.282, 0.302])
    colors = [GRAY, ORANGE, BLUE, CYAN, PURPLE, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.55))
    panels = [
        (axes[0], teds, "TEDS", (0.527, 0.547), "a"),
        (axes[1], acc, "Structure accuracy", (0.267, 0.307), "b"),
    ]
    offsets = {
        "Cell token": (-5, -13),
        "$\\lambda_{pres}=0$": (-8, -17),
        "Full model": (-7, 10),
        "No bidirectional": (-3, -15),
        "No geometry": (6, -2),
        "Aux off": (-4, 10),
    }

    for ax, score, ylabel, ylim, label in panels:
        ax.scatter(fps, score, s=72, c=colors, edgecolor="white", linewidth=0.9, zorder=3)
        ax.set_xlim(8.2, 21.2)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Inference speed (FPS)")
        ax.set_ylabel(ylabel)
        ax.grid(color=GRID, linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        add_panel_label(ax, label)
        for name, xval, yval in zip(names, fps, score):
            dx, dy = offsets[name]
            ax.annotate(
                name,
                (xval, yval),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.2,
                color=DARK,
            )

        ax.annotate(
            "Auxiliary branch removed at inference",
            xy=(18.68, score[-1]),
            xytext=(15.0, ylim[0] + (ylim[1] - ylim[0]) * 0.20),
            arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.1},
            color=GREEN,
            fontsize=7.4,
            ha="center",
        )

    fig.suptitle(
        "Ablation-based accuracy-efficiency trade-off on CTA500",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, output_dir, "ablation_2_accuracy_efficiency")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_figures_20260713"),
    )
    args = parser.parse_args()

    configure_style()
    comparison_general(args.output_dir)
    comparison_financial(args.output_dir)
    ablation_components(args.output_dir)
    ablation_tradeoff(args.output_dir)
    print(f"Generated figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
