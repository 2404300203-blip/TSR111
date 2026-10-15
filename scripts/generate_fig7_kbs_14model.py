#!/usr/bin/env python3
"""Generate a wide KBS-style qualitative plate for the repeated-span case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
DISPLAY_ASPECT = 0.24

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

KBS_GREEN = "#2A8C82"
KBS_LIGHT_GREEN = "#E4F2EF"
KBS_RED = "#C96868"
KBS_LIGHT_RED = "#F8E7E5"
KBS_MISSING = "#A66F9D"


def make_figure(
    project: Path,
    output_dir: Path,
    records_root: Path | None = None,
    image_path_override: Path | None = None,
) -> None:
    records_root = records_root or project / "output" / "main_tables_rebuild_20260810"
    data = {}
    for _, directory, _, _ in MODEL_SPECS:
        data[directory] = load_jsonl(records_root / directory / "sample_teds.jsonl")

    ours_record = data["dbm_slanet"][SAMPLE_INDEX]
    image_path = image_path_override or project / ours_record["image_path"]
    image = Image.open(image_path).convert("RGB")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    # Match the restrained teal/coral/plum palette used in KBS qualitative
    # figures while preserving the original correctness semantics.
    draw_structure.__globals__.update(
        {
            "GREEN": KBS_GREEN,
            "LIGHT_GREEN": KBS_LIGHT_GREEN,
            "RED": KBS_RED,
            "LIGHT_RED": KBS_LIGHT_RED,
            "ORANGE": KBS_MISSING,
        }
    )
    fig = plt.figure(figsize=(8.3, 6.65), dpi=180, facecolor="white")
    grid = fig.add_gridspec(
        4,
        4,
        left=0.045,
        right=0.975,
        top=0.978,
        bottom=0.035,
        hspace=0.12,
        wspace=0.17,
    )

    # Input, ground truth, and all 14 predictions occupy equal grid cells.
    image_ax = fig.add_subplot(grid[0, 0])
    image_ax.imshow(image)
    image_ax.axis("off")
    image_ax.set_title("Input table image", fontsize=8.0, fontweight="bold", color=DARK, pad=8)
    gt_ax = fig.add_subplot(grid[0, 1])
    draw_structure(gt_ax, ours_record["gt_tokens"], "Ground truth", None)
    gt_ax.set_aspect(DISPLAY_ASPECT, adjustable="box")
    for text in gt_ax.texts:
        text.set_fontsize(4.2)
    gt_ax.title.set_fontsize(8.0)
    gt_ax.title.set_color(DARK)

    # Fill the remaining cells in reading order; ours lands at bottom right.
    letters = "abcdefghijklmn"
    positions = [(0, 2), (0, 3)] + [(row, col) for row in range(1, 4) for col in range(4)]
    for spec, letter, (row, col) in zip(MODEL_SPECS, letters, positions):
        name, directory, display, family = spec
        record = data[directory][SAMPLE_INDEX]
        ax = fig.add_subplot(grid[row, col])
        draw_structure(ax, record["pred_tokens"], f"({letter}) {display}", None, ours_record["gt_tokens"])
        ax.set_aspect(DISPLAY_ASPECT, adjustable="box")
        for text in ax.texts:
            text.set_fontsize(4.2)
        ax.title.set_fontsize(8.0)
        ax.title.set_linespacing(0.98)
        ax.title.set_color(FAMILY_COLORS[family])
        ax.title.set_fontweight("bold" if family == "ours" else "normal")

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(output_dir / f"fig7_kbs_14model.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--image-path", type=Path)
    args = parser.parse_args()
    make_figure(
        args.project_root.resolve(),
        args.output_dir.resolve(),
        args.records_root.resolve() if args.records_root else None,
        args.image_path.resolve() if args.image_path else None,
    )
    print(f"Generated {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
