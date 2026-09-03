#!/usr/bin/env python3
"""Generate sharp KBS-style A/B comparisons from real prediction JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from generate_real_qualitative_effect_figures import load_jsonl
from generate_six_model_kbs_optimized import (
    INK,
    OURS_COLOR,
    draw_normalized_structure,
)


MODEL_FILES = [
    (
        "SLANet",
        "output/phase10_large_val_20260624/baseline_val5000_sample_teds.jsonl",
    ),
    (
        "Cell Token",
        "output/phase10_more_models_val5000_20260710/e11a_cell_token_sample_teds.jsonl",
    ),
    (
        "Lite Mamba",
        "output/phase10_more_models_val5000_20260710/e4_lite_stage7_sample_teds.jsonl",
    ),
    (
        "Horizontal Mamba",
        "output/phase10_axis_ablation_val5000_20260710/expC_horizontal_val5000_sample_teds.jsonl",
    ),
    (
        "DBM-SLANet (Ours)",
        "output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl",
    ),
]

CASES = [
    (4313, "a", "qualitative_comparison_a_kbs_vector"),
    (1158, "b", "qualitative_comparison_b_kbs_vector"),
]

GT_FILL = "#E8EFF7"


def harmonize_input(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    colorized = ImageOps.colorize(grayscale, black=INK, white=GT_FILL)
    scale = max(1, int(np.ceil(6000 / colorized.width)))
    return colorized.resize(
        (colorized.width * scale, colorized.height * scale),
        Image.Resampling.NEAREST,
    )


def draw_left_title(
    ax: plt.Axes,
    title: str,
    color: str = INK,
    fontsize: float = 9.2,
) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        0.5,
        title,
        ha="left",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
    )


def make_case(
    project: Path,
    output_dir: Path,
    sample_index: int,
    panel_label: str,
    stem: str,
    model_data: list[tuple[str, dict[int, dict]]],
) -> None:
    ours_record = model_data[-1][1][sample_index]
    image_path = project / ours_record["image_path"]
    with Image.open(image_path) as source:
        input_image = harmonize_input(source.convert("RGB"))
        input_aspect = source.width / source.height

    scores = [records[sample_index]["teds"] for _, records in model_data]
    if scores[-1] <= max(scores[:-1]):
        raise ValueError(
            f"Sample {sample_index}: ours={scores[-1]:.4f}, "
            f"best_other={max(scores[:-1]):.4f}"
        )

    structure_aspect = min(2.15, max(1.35, input_aspect))
    header_height = min(4.55, max(2.2, 8.5 / input_aspect))
    panel_height = 2.72 / structure_aspect
    figure_height = header_height + 2 * (panel_height + 0.42) + 0.75

    fig = plt.figure(figsize=(9.0, figure_height), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.022,
        right=0.985,
        bottom=0.03,
        top=0.98,
        height_ratios=[header_height, 2 * (panel_height + 0.34)],
        hspace=0.15,
    )

    header = outer[0].subgridspec(
        2, 1, height_ratios=[0.12, 1.0], hspace=0.015
    )
    header_title_ax = fig.add_subplot(header[0])
    header_title_ax.axis("off")
    header_title_ax.text(
        0.0,
        0.5,
        f"({panel_label})",
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    header_title_ax.text(
        0.5,
        0.5,
        "Input table image",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    input_ax = fig.add_subplot(header[1])
    input_ax.imshow(input_image, interpolation="nearest")
    input_ax.axis("off")

    grid = outer[1].subgridspec(2, 3, hspace=0.28, wspace=0.08)
    panels = [("Ground Truth", ours_record["gt_tokens"], None)]
    panels.extend(
        (name, records[sample_index]["pred_tokens"], ours_record["gt_tokens"])
        for name, records in model_data
    )

    for index, (title, tokens, gt_tokens) in enumerate(panels):
        slot = grid[index // 3, index % 3].subgridspec(
            2, 1, height_ratios=[0.16, 1.0], hspace=0.015
        )
        title_ax = fig.add_subplot(slot[0])
        draw_left_title(
            title_ax,
            title,
            OURS_COLOR if title == "DBM-SLANet (Ours)" else INK,
        )
        structure_ax = fig.add_subplot(slot[1])
        draw_normalized_structure(
            structure_ax,
            tokens,
            "",
            structure_aspect,
            gt_tokens,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    model_data = [(name, load_jsonl(project / path)) for name, path in MODEL_FILES]
    for sample_index, panel_label, stem in CASES:
        make_case(
            project,
            args.output_dir,
            sample_index,
            panel_label,
            stem,
            model_data,
        )
    print(f"Generated sharp KBS A/B figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
