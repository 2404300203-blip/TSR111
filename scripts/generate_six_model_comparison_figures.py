#!/usr/bin/env python3
"""Generate four real qualitative comparisons across six table models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from generate_real_qualitative_effect_figures import (
    DARK,
    GREEN,
    draw_structure,
    load_jsonl,
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
        "Vertical Mamba",
        "output/phase10_axis_ablation_val5000_20260710/expD_vertical_val5000_sample_teds.jsonl",
    ),
    (
        "DBM-SLANet (Ours)",
        "output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl",
    ),
]


CASES = [
    (2436, "comparison_case_1_six_models"),
    (4477, "comparison_case_2_six_models"),
    (1158, "comparison_case_3_six_models"),
    (4372, "comparison_case_4_six_models"),
]


def make_case_figure(
    project: Path,
    output_dir: Path,
    sample_index: int,
    stem: str,
    model_data: list[tuple[str, dict[int, dict]]],
) -> None:
    ours_record = model_data[-1][1][sample_index]
    image_path = project / ours_record["image_path"]
    with Image.open(image_path) as source:
        aspect = source.width / source.height

    header_height = min(3.8, max(2.15, 4.9 / aspect))
    figure_height = header_height + 5.9
    fig = plt.figure(figsize=(7.2, figure_height), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.025,
        right=0.99,
        bottom=0.028,
        top=0.985,
        height_ratios=[header_height, 5.35],
        hspace=0.13,
    )

    header = outer[0].subgridspec(1, 2, width_ratios=[1.68, 1.0], wspace=0.07)
    image_ax = fig.add_subplot(header[0, 0])
    image_ax.imshow(Image.open(image_path).convert("RGB"))
    image_ax.axis("off")
    image_ax.set_title("Input table image", fontsize=9.5, fontweight="bold", color=DARK, pad=3)

    gt_ax = fig.add_subplot(header[0, 1])
    draw_structure(gt_ax, ours_record["gt_tokens"], "Ground truth", None)
    gt_ax.title.set_fontsize(9.5)

    prediction_grid = outer[1].subgridspec(2, 3, hspace=0.34, wspace=0.035)
    panel_labels = ["a", "b", "c", "d", "e", "f"]
    scores = []
    for idx, ((model_name, records), label) in enumerate(zip(model_data, panel_labels)):
        record = records[sample_index]
        scores.append(record["teds"])
        ax = fig.add_subplot(prediction_grid[idx // 3, idx % 3])
        draw_structure(
            ax,
            record["pred_tokens"],
            f"({label}) {model_name}",
            record["teds"],
            ours_record["gt_tokens"],
        )
        ax.title.set_fontsize(8.4)
        if idx == len(model_data) - 1:
            ax.title.set_color(GREEN)
            ax.title.set_fontweight("bold")

    best_other = max(scores[:-1])
    if scores[-1] <= best_other:
        raise ValueError(
            f"Sample {sample_index} does not support the intended conclusion: "
            f"ours={scores[-1]:.4f}, best_other={best_other:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    model_data = [(name, load_jsonl(project / path)) for name, path in MODEL_FILES]

    for sample_index, stem in CASES:
        make_case_figure(project, args.output_dir, sample_index, stem, model_data)

    print(f"Generated six-model comparison figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
