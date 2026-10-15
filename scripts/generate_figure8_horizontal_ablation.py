#!/usr/bin/env python3
"""Generate horizontal four-way qualitative ablation panels for Figure 8."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from generate_real_qualitative_effect_figures import (
    BLUE,
    DARK,
    GREEN,
    LIGHT_BLUE,
    LIGHT_GREEN,
    LIGHT_RED,
    ORANGE,
    RED,
    tokens_to_cells,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "论文材料" / "figure8_horizontal_4way_20260730"
INPUTS = OUTPUT / "source_inputs"
RECORDS = ROOT / "remote_results" / "figure8_horizontal_records.json"


def load_records() -> dict[str, dict[str, dict]]:
    with RECORDS.open(encoding="utf-8") as handle:
        return json.load(handle)


def sharpened_table(path: Path) -> Image.Image:
    image = ImageOps.grayscale(Image.open(path))
    image = ImageOps.autocontrast(image, cutoff=0.5)
    image = image.resize((image.width * 8, image.height * 8), Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.35)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=220, threshold=1))
    return image.convert("RGB")


def draw_input(ax: plt.Axes, image_path: Path) -> None:
    image = sharpened_table(image_path)
    ax.imshow(image, interpolation="none", resample=False)
    ax.axis("off")


def draw_horizontal_structure(
    ax: plt.Axes,
    tokens: list[str],
    title: str,
    score: float | None,
    target_aspect: float,
    gt_tokens: list[str] | None = None,
) -> None:
    cells, nrows, ncols = tokens_to_cells(tokens)
    gt_cells: set[tuple[int, int, int, int]] | None = None
    if gt_tokens is not None:
        gt_list, gt_rows, gt_cols = tokens_to_cells(gt_tokens)
        gt_cells = set(gt_list)
        nrows = max(nrows, gt_rows)
        ncols = max(ncols, gt_cols)

    x_scale = target_aspect * nrows / ncols
    pred_set = set(cells)

    for row, col, rowspan, colspan in cells:
        x = col * x_scale
        y = nrows - row - rowspan
        if gt_cells is None:
            edge, face = BLUE, LIGHT_BLUE
        else:
            correct = (row, col, rowspan, colspan) in gt_cells
            edge = GREEN if correct else RED
            face = LIGHT_GREEN if correct else LIGHT_RED
        ax.add_patch(
            Rectangle(
                (x, y),
                colspan * x_scale,
                rowspan,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.55,
            )
        )
        if rowspan > 1 or colspan > 1:
            ax.text(
                x + colspan * x_scale / 2,
                y + rowspan / 2,
                f"{rowspan}x{colspan}",
                ha="center",
                va="center",
                fontsize=7.1,
                color=edge,
                fontweight="bold",
            )

    if gt_cells is not None:
        for row, col, rowspan, colspan in sorted(gt_cells - pred_set):
            x = col * x_scale
            y = nrows - row - rowspan
            inset_x = min(0.08 * x_scale, 0.08)
            ax.add_patch(
                Rectangle(
                    (x + inset_x, y + 0.04),
                    max(0.05, colspan * x_scale - 2 * inset_x),
                    max(0.05, rowspan - 0.08),
                    fill=False,
                    edgecolor=ORANGE,
                    linewidth=1.65,
                    linestyle=(0, (4, 2)),
                    zorder=5,
                )
            )

    width = ncols * x_scale
    ax.set_xlim(-0.015 * width, width * 1.015)
    ax.set_ylim(-0.12, nrows + 0.12)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    subtitle = title if score is None else f"{title}\nTEDS = {score:.3f}"
    ax.set_title(subtitle, fontsize=10.0, fontweight="bold", color=DARK, pad=6)


def draw_case(
    fig: plt.Figure,
    spec,
    panel_label: str,
    panel_title: str,
    image_path: Path,
    ground_truth: dict,
    lite_mamba: dict,
    ablated: dict,
    ablated_title: str,
    full_model: dict,
) -> None:
    frame = fig.add_subplot(spec)
    frame.axis("off")
    frame.text(
        0.0,
        1.025,
        f"({panel_label}) {panel_title}",
        transform=frame.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=DARK,
    )

    inner = GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=spec,
        height_ratios=[2.45, 1.0, 1.0],
        hspace=0.28,
        wspace=0.14,
    )

    image_ax = fig.add_subplot(inner[0, :])
    draw_input(image_ax, image_path)
    with Image.open(image_path) as source:
        target_aspect = min(2.9, max(2.3, source.width / source.height))

    axes = [
        fig.add_subplot(inner[1, 0]),
        fig.add_subplot(inner[1, 1]),
        fig.add_subplot(inner[2, 0]),
        fig.add_subplot(inner[2, 1]),
    ]
    gt_tokens = ground_truth["gt_tokens"]
    draw_horizontal_structure(axes[0], gt_tokens, "Ground truth", None, target_aspect)
    draw_horizontal_structure(
        axes[1],
        lite_mamba["pred_tokens"],
        "Lite Mamba",
        lite_mamba["teds"],
        target_aspect,
        gt_tokens,
    )
    draw_horizontal_structure(
        axes[2],
        ablated["pred_tokens"],
        ablated_title,
        ablated["teds"],
        target_aspect,
        gt_tokens,
    )
    draw_horizontal_structure(
        axes[3],
        full_model["pred_tokens"],
        "Full model",
        full_model["teds"],
        target_aspect,
        gt_tokens,
    )

    for ax in axes:
        ax.title.set_fontsize(10.0)
        ax.title.set_fontweight("bold")
        ax.title.set_y(0.995)
        ax.set_anchor("N")


def add_legend(fig: plt.Figure, y: float, fontsize: float = 8.5) -> None:
    legend = [
        Patch(facecolor=LIGHT_GREEN, edgecolor=GREEN, label="Correct predicted cell"),
        Patch(facecolor=LIGHT_RED, edgecolor=RED, label="Incorrect predicted cell"),
        Patch(facecolor="white", edgecolor=ORANGE, linestyle="--", label="Missing GT cell"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=3,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.8,
        columnspacing=1.6,
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in (
        ("png", {"dpi": 800}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(
            OUTPUT / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.08,
            **kwargs,
        )


def make_standalone(
    records: dict[str, dict[str, dict]],
    panel_label: str,
    panel_title: str,
    sample_idx: int,
    ablation_key: str,
    ablation_title: str,
    stem: str,
) -> None:
    key = str(sample_idx)
    full_model = records["Full model"][key]
    fig = plt.figure(figsize=(11.4, 8.0), facecolor="white")
    spec = fig.add_gridspec(
        1,
        1,
        left=0.035,
        right=0.985,
        bottom=0.035,
        top=0.92,
    )[0, 0]
    draw_case(
        fig,
        spec,
        panel_label,
        panel_title,
        INPUTS / full_model["filename"],
        full_model,
        records["Lite Mamba"][key],
        records[ablation_key][key],
        ablation_title,
        full_model,
    )
    save_all(fig, stem)
    plt.close(fig)


def make_combined(records: dict[str, dict[str, dict]]) -> None:
    fig = plt.figure(figsize=(11.4, 15.2), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        left=0.035,
        right=0.985,
        bottom=0.035,
        top=0.96,
        hspace=0.16,
    )

    cases = [
        (
            outer[0, 0],
            "c",
            "Bidirectional scan ablation",
            48,
            "No bidirectional",
            "No bidirectional",
        ),
        (
            outer[1, 0],
            "d",
            "Geometry input ablation",
            4052,
            "No geometry",
            "No geometry",
        ),
    ]
    for spec, label, title, sample_idx, ablation_key, ablation_title in cases:
        key = str(sample_idx)
        full_model = records["Full model"][key]
        draw_case(
            fig,
            spec,
            label,
            title,
            INPUTS / full_model["filename"],
            full_model,
            records["Lite Mamba"][key],
            records[ablation_key][key],
            ablation_title,
            full_model,
        )

    save_all(fig, "figure8_cd_2x2_vertical_combined_sharp")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    make_standalone(
        records,
        "c",
        "Bidirectional scan ablation",
        48,
        "No bidirectional",
        "No bidirectional",
        "figure8_c_bidirectional_2x2_sharp",
    )
    make_standalone(
        records,
        "d",
        "Geometry input ablation",
        4052,
        "No geometry",
        "No geometry",
        "figure8_d_geometry_2x2_sharp",
    )
    make_combined(records)


if __name__ == "__main__":
    main()
