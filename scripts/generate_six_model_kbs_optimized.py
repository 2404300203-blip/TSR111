#!/usr/bin/env python3
"""Generate an aspect-aligned six-model comparison in a KBS-style palette."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

from generate_real_qualitative_effect_figures import load_jsonl, tokens_to_cells


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
        "Vertical Mamba",
        "output/phase10_axis_ablation_val5000_20260710/expD_vertical_val5000_sample_teds.jsonl",
    ),
    (
        "DBM-SLANet (Ours)",
        "output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl",
    ),
]

SAMPLE_INDEX = 2436

# Muted, colorblind-friendly academic palette commonly used in KBS-style figures.
INK = "#25364A"
GT_EDGE = "#4472A4"
GT_FILL = "#E8EFF7"
CORRECT_EDGE = "#2A7F74"
CORRECT_FILL = "#E2F0EC"
ERROR_EDGE = "#C65D3A"
ERROR_FILL = "#F7E7E0"
MISSING_EDGE = "#B07AA1"
SPAN_LINE = "#7B8794"
OURS_COLOR = "#2A7F74"


def add_span_guides(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    rowspan: int,
    colspan: int,
) -> None:
    """Show the implicit grid inside merged cells with dashed separators."""
    for index in range(1, colspan):
        guide_x = x + width * index / colspan
        ax.plot(
            [guide_x, guide_x],
            [y, y + height],
            color=SPAN_LINE,
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            alpha=0.9,
            zorder=4,
        )
    for index in range(1, rowspan):
        guide_y = y + height * index / rowspan
        ax.plot(
            [x, x + width],
            [guide_y, guide_y],
            color=SPAN_LINE,
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            alpha=0.9,
            zorder=4,
        )


def draw_normalized_structure(
    ax: plt.Axes,
    tokens: list[str],
    title: str,
    target_aspect: float,
    gt_tokens: list[str] | None = None,
) -> None:
    cells, pred_rows, pred_cols = tokens_to_cells(tokens)
    gt_cells: set[tuple[int, int, int, int]] | None = None
    gt_rows = pred_rows
    gt_cols = pred_cols
    if gt_tokens is not None:
        parsed_gt, gt_rows, gt_cols = tokens_to_cells(gt_tokens)
        gt_cells = set(parsed_gt)

    # Every panel occupies the same target rectangle. Predictions with different
    # row/column counts are normalized into this shared outer shape.
    canvas_rows = max(pred_rows, gt_rows, 1)
    canvas_cols = max(pred_cols, gt_cols, 1)
    scale_x = target_aspect / canvas_cols
    scale_y = 1.0 / canvas_rows
    pred_set = set(cells)

    for row, col, rowspan, colspan in cells:
        correct = gt_cells is None or (row, col, rowspan, colspan) in gt_cells
        edge = GT_EDGE if gt_cells is None else (CORRECT_EDGE if correct else ERROR_EDGE)
        face = GT_FILL if gt_cells is None else (CORRECT_FILL if correct else ERROR_FILL)
        linestyle = "-" if correct else (0, (4, 2))
        x = col * scale_x
        y = (canvas_rows - row - rowspan) * scale_y
        width = colspan * scale_x
        height = rowspan * scale_y
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.15,
                linestyle=linestyle,
                zorder=2,
            )
        )
        if rowspan > 1 or colspan > 1:
            add_span_guides(ax, x, y, width, height, rowspan, colspan)
            ax.text(
                x + width / 2,
                y + height / 2,
                f"{rowspan}x{colspan}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=edge,
                fontweight="bold",
                zorder=5,
            )

    if gt_cells is not None:
        for row, col, rowspan, colspan in sorted(gt_cells - pred_set):
            x = col * scale_x + 0.025 * scale_x
            y = (canvas_rows - row - rowspan) * scale_y + 0.025 * scale_y
            width = max(0.03, colspan * scale_x - 0.05 * scale_x)
            height = max(0.03, rowspan * scale_y - 0.05 * scale_y)
            ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    fill=False,
                    edgecolor=MISSING_EDGE,
                    linewidth=1.15,
                    linestyle=(0, (4, 2)),
                    zorder=6,
                )
            )

    ax.set_xlim(-0.02 * target_aspect, 1.02 * target_aspect)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=9.2, fontweight="bold", color=INK, pad=5)


def make_figure(
    project: Path,
    output_dir: Path,
    model_data: list[tuple[str, dict[int, dict]]],
) -> None:
    ours_record = model_data[-1][1][SAMPLE_INDEX]
    image_path = project / ours_record["image_path"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        input_aspect = source.width / source.height

    # Harmonize the real input with the light blue-gray Ground Truth palette while
    # retaining all original text and line content.
    grayscale = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
    ink_strength = 1.0 / (1.0 + np.exp((grayscale - 158.0) / 9.0))
    ink_rgb = np.asarray(Image.new("RGB", (1, 1), INK), dtype=np.float32)[0, 0]
    fill_rgb = np.asarray(Image.new("RGB", (1, 1), GT_FILL), dtype=np.float32)[0, 0]
    harmonized_array = (
        fill_rgb[None, None, :] * (1.0 - ink_strength[:, :, None])
        + ink_rgb[None, None, :] * ink_strength[:, :, None]
    )
    harmonized_image = Image.fromarray(
        np.clip(harmonized_array, 0, 255).astype(np.uint8), mode="RGB"
    )

    scores = [records[SAMPLE_INDEX]["teds"] for _, records in model_data]
    if scores[-1] <= max(scores[:-1]):
        raise ValueError("Selected sample no longer supports the intended conclusion")

    # Limit extremely wide inputs while preserving their overall visual character.
    structure_aspect = min(2.15, max(1.35, input_aspect))
    fig = plt.figure(figsize=(9.0, 7.25), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.025,
        right=0.985,
        bottom=0.035,
        top=0.975,
        height_ratios=[1.15, 2.0],
        hspace=0.16,
    )

    header = outer[0].subgridspec(
        2,
        2,
        width_ratios=[1.7, 1.0],
        height_ratios=[0.12, 1.0],
        wspace=0.11,
        hspace=0.015,
    )
    input_title_ax = fig.add_subplot(header[0, 0])
    gt_title_ax = fig.add_subplot(header[0, 1])
    for title_ax, title in (
        (input_title_ax, "Input table image"),
        (gt_title_ax, "Ground truth"),
    ):
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.5,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )

    image_ax = fig.add_subplot(header[1, 0])
    image_ax.imshow(harmonized_image)
    image_ax.axis("off")

    gt_ax = fig.add_subplot(header[1, 1])
    draw_normalized_structure(
        gt_ax,
        ours_record["gt_tokens"],
        "",
        structure_aspect,
    )

    prediction_grid = outer[1].subgridspec(2, 3, hspace=0.25, wspace=0.08)
    panel_labels = "abcdef"
    for index, ((model_name, records), panel_label) in enumerate(
        zip(model_data, panel_labels)
    ):
        slot = prediction_grid[index // 3, index % 3].subgridspec(
            2, 1, height_ratios=[0.12, 1.0], hspace=0.015
        )
        title_ax = fig.add_subplot(slot[0])
        title_ax.axis("off")
        title_ax.text(
            0.0,
            0.5,
            f"({panel_label}) {model_name}",
            ha="left",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=OURS_COLOR if index == len(model_data) - 1 else INK,
        )
        ax = fig.add_subplot(slot[1])
        draw_normalized_structure(
            ax,
            records[SAMPLE_INDEX]["pred_tokens"],
            "",
            structure_aspect,
            ours_record["gt_tokens"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "comparison_case_1_kbs_optimized"
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
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
    make_figure(project, args.output_dir, model_data)
    print(f"Generated KBS-style comparison in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
