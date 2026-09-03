#!/usr/bin/env python3
"""Generate two compact real qualitative figures with five compared models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from generate_real_qualitative_effect_figures import DARK, GREEN, draw_structure, load_jsonl


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
    (4372, "a", "qualitative_comparison_a"),
    (1158, "b", "qualitative_comparison_b_complex"),
]


def make_case_figure(
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
        source_rgb = source.convert("RGB")
        aspect = source.width / source.height

    scores = [records[sample_index]["teds"] for _, records in model_data]
    if scores[-1] <= max(scores[:-1]):
        raise ValueError(
            f"Sample {sample_index} does not support the intended comparison: "
            f"ours={scores[-1]:.4f}, best_other={max(scores[:-1]):.4f}"
        )

    # Reserve enough height for the real image without stretching it. The structure
    # grid stays fixed so A and the more complex B have consistent visual scale.
    image_height = min(3.45, max(2.05, 7.9 / aspect))
    figure_height = image_height + 7.05
    # A narrower page keeps the six comparison panels visually close together.
    fig = plt.figure(figsize=(7.2, figure_height), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.025,
        right=0.985,
        bottom=0.025,
        top=0.975,
        height_ratios=[image_height, 6.45],
        hspace=0.12,
    )

    image_ax = fig.add_subplot(outer[0])
    image_ax.imshow(source_rgb)
    image_ax.axis("off")
    image_ax.text(
        0.0,
        1.01,
        f"({panel_label})",
        transform=image_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=DARK,
    )

    structure_grid = outer[1].subgridspec(2, 3, hspace=0.20, wspace=0.0)
    panels = [("Ground Truth", ours_record["gt_tokens"], None)]
    panels.extend(
        (name, records[sample_index]["pred_tokens"], ours_record["gt_tokens"])
        for name, records in model_data
    )

    for idx, (title, tokens, gt_tokens) in enumerate(panels):
        ax = fig.add_subplot(structure_grid[idx // 3, idx % 3])
        draw_structure(ax, tokens, title, None, gt_tokens)
        ax.title.set_fontsize(9.2)
        if title == "DBM-SLANet (Ours)":
            ax.title.set_color(GREEN)
            ax.title.set_fontweight("bold")

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
        make_case_figure(
            project,
            args.output_dir,
            sample_index,
            panel_label,
            stem,
            model_data,
        )

    print(f"Generated five-model A/B figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
