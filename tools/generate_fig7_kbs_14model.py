#!/usr/bin/env python3
"""Generate a wide KBS-style qualitative plate for the repeated-span case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

# Allow the script to run from the repository's tools directory while reusing
# the established qualitative renderer in the sibling scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_real_qualitative_effect_figures import (
    DARK,
    GREEN,
    LIGHT_GREEN,
    LIGHT_RED,
    ORANGE,
    RED,
    draw_structure,
    load_jsonl,
)


SAMPLE_INDEX = 2436

MODEL_SPECS = [
    ("SLANet", "slanet", "SLANet", "baseline"),
    ("Cell Token", "cell_token", "Cell Token", "token"),
    ("Lite Mamba", "lite_mamba", "Lite Mamba", "lite"),
    ("Full Mamba, no presence", "full_mamba_no_presence", "Mamba\n(no presence)", "ablation"),
    ("Full Mamba, presence", "full_mamba_presence", "Mamba\n(presence)", "ablation"),
    ("Full Mamba, one-way", "full_mamba_one_way", "Mamba\n(one-way)", "ablation"),
    ("True Horizontal-only", "true_horizontal", "Horizontal-only", "axis"),
    ("True Vertical-only", "true_vertical", "Vertical-only", "axis"),
    ("Full Mamba, zero geometry", "zero_geometry", "Mamba\n(zero geometry)", "ablation"),
    ("Full DBM, d=64", "dbm_d64", "DBM\n(d=64)", "ablation"),
    ("Full DBM, d_s=8", "dbm_ds8", "DBM\n(d_s=8)", "ablation"),
    ("Full DBM, d=192", "dbm_d192", "DBM\n(d=192)", "ablation"),
    ("Full DBM, e=1", "dbm_e1", "DBM\n(e=1)", "ablation"),
    ("DBM-SLANet (Ours)", "dbm_slanet", "DBM-SLANet (ours)", "ours"),
]

FAMILY_COLORS = {
    "baseline": "#376DA3",
    "token": "#C78B2E",
    "lite": "#B56A2A",
    "ablation": "#667C92",
    "axis": "#4B8F6A",
    "ours": "#138A83",
}


def make_figure(project: Path, output_dir: Path) -> None:
    data = {}
    for _, directory, _, _ in MODEL_SPECS:
        data[directory] = load_jsonl(
            project / "output" / "main_tables_rebuild_20260810" / directory / "sample_teds.jsonl"
        )

    ours_record = data["dbm_slanet"][SAMPLE_INDEX]
    image_path = project / ours_record["image_path"]
    image = Image.open(image_path).convert("RGB")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(8.3, 11.35), dpi=180, facecolor="white")
    outer = fig.add_gridspec(
        5,
        1,
        left=0.045,
        right=0.975,
        top=0.978,
        bottom=0.075,
        height_ratios=[2.05, 2.12, 2.12, 2.12, 2.12],
        hspace=0.34,
    )

    # Header: the real input and its topology are the visual anchor.
    header = outer[0].subgridspec(1, 2, width_ratios=[1.54, 1.0], wspace=0.09)
    image_ax = fig.add_subplot(header[0, 0])
    image_ax.imshow(image)
    image_ax.axis("off")
    image_ax.set_title("Input table image", fontsize=12, fontweight="bold", color=DARK, pad=7)
    gt_ax = fig.add_subplot(header[0, 1])
    draw_structure(gt_ax, ours_record["gt_tokens"], "Ground-truth topology", None)
    gt_ax.title.set_fontsize(12)
    gt_ax.title.set_color(DARK)

    # The 14 tested variants form a full-page four-column comparison matrix.
    letters = "abcdefghijklmn"
    grids = [outer[row + 1].subgridspec(1, 4, wspace=0.17) for row in range(4)]
    positions = [(row, col) for row in range(3) for col in range(4)] + [(3, 1), (3, 2)]
    for spec, letter, (row, col) in zip(MODEL_SPECS, letters, positions):
        name, directory, display, family = spec
        record = data[directory][SAMPLE_INDEX]
        ax = fig.add_subplot(grids[row][0, col])
        draw_structure(ax, record["pred_tokens"], f"({letter}) {display}", record["teds"], ours_record["gt_tokens"])
        ax.title.set_fontsize(8.0)
        ax.title.set_linespacing(0.98)
        ax.title.set_color(FAMILY_COLORS[family])
        ax.title.set_fontweight("bold" if family == "ours" else "normal")

    fig.text(
        0.045,
        0.016,
        "Same CTA-500 complex case (index 2436) across all panels.",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#52606D",
    )
    legend = [
        Patch(facecolor=LIGHT_GREEN, edgecolor=GREEN, label="Correct"),
        Patch(facecolor=LIGHT_RED, edgecolor=RED, label="Incorrect"),
        Patch(facecolor="white", edgecolor=ORANGE, linestyle="--", label="Missing GT span"),
    ]
    fig.legend(handles=legend, loc="lower right", bbox_to_anchor=(0.975, 0.012), ncol=3, frameon=False, fontsize=7.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(output_dir / f"fig7_kbs_14model.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.project_root.resolve(), args.output_dir.resolve())
    print(f"Generated {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
