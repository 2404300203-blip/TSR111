#!/usr/bin/env python3
"""Render real table images and model structure predictions as qualitative figures."""

from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

BLUE = "#3D6FA8"
LIGHT_BLUE = "#EAF1F8"
GREEN = "#4D8D68"
LIGHT_GREEN = "#E8F4EC"
RED = "#B95450"
LIGHT_RED = "#FAEAE8"
ORANGE = "#D68A2E"
DARK = "#263442"
GRAY = "#66717E"


class TableStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[tuple[int, int]]] = []
        self.current_row: list[tuple[int, int]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"} and self.current_row is not None:
            attr_map = dict(attrs)
            try:
                rowspan = max(1, int(attr_map.get("rowspan") or 1))
            except ValueError:
                rowspan = 1
            try:
                colspan = max(1, int(attr_map.get("colspan") or 1))
            except ValueError:
                colspan = 1
            self.current_row.append((rowspan, colspan))

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None


def tokens_to_cells(tokens: list[str]) -> tuple[list[tuple[int, int, int, int]], int, int]:
    parser = TableStructureParser()
    parser.feed("".join(tokens))

    occupied: set[tuple[int, int]] = set()
    cells: list[tuple[int, int, int, int]] = []
    ncols = 0
    nrows = len(parser.rows)

    for row_idx, row_cells in enumerate(parser.rows):
        col_idx = 0
        for rowspan, colspan in row_cells:
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            cells.append((row_idx, col_idx, rowspan, colspan))
            for rr in range(row_idx, row_idx + rowspan):
                for cc in range(col_idx, col_idx + colspan):
                    occupied.add((rr, cc))
            nrows = max(nrows, row_idx + rowspan)
            ncols = max(ncols, col_idx + colspan)
            col_idx += colspan

    return cells, max(nrows, 1), max(ncols, 1)


def load_jsonl(path: Path) -> dict[int, dict]:
    records = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            records[record["sample_index"]] = record
    return records


def draw_structure(
    ax: plt.Axes,
    tokens: list[str],
    title: str,
    score: float | None,
    gt_tokens: list[str] | None = None,
) -> None:
    cells, nrows, ncols = tokens_to_cells(tokens)
    gt_cells = None
    if gt_tokens is not None:
        gt_cells_list, gt_rows, gt_cols = tokens_to_cells(gt_tokens)
        gt_cells = set(gt_cells_list)
        nrows = max(nrows, gt_rows)
        ncols = max(ncols, gt_cols)

    pred_set = set(cells)
    if gt_cells is None:
        for row, col, rowspan, colspan in cells:
            y = nrows - row - rowspan
            ax.add_patch(
                Rectangle(
                    (col, y),
                    colspan,
                    rowspan,
                    facecolor=LIGHT_BLUE,
                    edgecolor=BLUE,
                    linewidth=1.25,
                )
            )
            if rowspan > 1 or colspan > 1:
                ax.text(
                    col + colspan / 2,
                    y + rowspan / 2,
                    f"{rowspan}x{colspan}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=BLUE,
                    fontweight="bold",
                )
    else:
        for row, col, rowspan, colspan in cells:
            is_correct = (row, col, rowspan, colspan) in gt_cells
            y = nrows - row - rowspan
            edge = GREEN if is_correct else RED
            face = LIGHT_GREEN if is_correct else LIGHT_RED
            ax.add_patch(
                Rectangle(
                    (col, y),
                    colspan,
                    rowspan,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.35,
                )
            )
            if rowspan > 1 or colspan > 1:
                ax.text(
                    col + colspan / 2,
                    y + rowspan / 2,
                    f"{rowspan}x{colspan}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=edge,
                    fontweight="bold",
                )

        for row, col, rowspan, colspan in sorted(gt_cells - pred_set):
            y = nrows - row - rowspan
            ax.add_patch(
                Rectangle(
                    (col + 0.04, y + 0.04),
                    max(0.05, colspan - 0.08),
                    max(0.05, rowspan - 0.08),
                    fill=False,
                    edgecolor=ORANGE,
                    linewidth=1.4,
                    linestyle=(0, (4, 2)),
                    zorder=5,
                )
            )

    ax.set_xlim(-0.15, ncols + 0.15)
    ax.set_ylim(-0.15, nrows + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    subtitle = title if score is None else f"{title}\nTEDS = {score:.3f}"
    ax.set_title(subtitle, fontsize=10.5, fontweight="bold", color=DARK, pad=8)


def draw_original(ax: plt.Axes, image_path: Path, filename: str) -> None:
    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"Real table image\n{filename}", fontsize=10.5, fontweight="bold", color=DARK, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B8C0C9")
        spine.set_linewidth(0.8)


def make_figure(
    output_dir: Path,
    stem: str,
    figure_title: str,
    original_path: Path,
    filename: str,
    gt_record: dict,
    left_record: dict,
    left_title: str,
    right_record: dict,
    right_title: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 6.6),
        gridspec_kw={"width_ratios": [1.25, 1], "height_ratios": [1, 1]},
    )
    draw_original(axes[0, 0], original_path, filename)
    draw_structure(axes[0, 1], gt_record["gt_tokens"], "Ground truth", None)
    draw_structure(
        axes[1, 0],
        left_record["pred_tokens"],
        left_title,
        left_record["teds"],
        gt_record["gt_tokens"],
    )
    draw_structure(
        axes[1, 1],
        right_record["pred_tokens"],
        right_title,
        right_record["teds"],
        gt_record["gt_tokens"],
    )

    fig.suptitle(figure_title, fontsize=13, fontweight="bold", color=DARK, y=0.98)
    legend = [
        Patch(facecolor=LIGHT_GREEN, edgecolor=GREEN, label="Correct predicted cell"),
        Patch(facecolor=LIGHT_RED, edgecolor=RED, label="Incorrect predicted cell"),
        Patch(facecolor="white", edgecolor=ORANGE, linestyle="--", label="Missing GT cell"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.02, 0.075, 0.98, 0.92), w_pad=1.6, h_pad=1.4)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_compact_case_panel(
    fig: plt.Figure,
    subplot_spec,
    panel_label: str,
    panel_title: str,
    original_path: Path,
    filename: str,
    gt_record: dict,
    left_record: dict,
    left_title: str,
    right_record: dict,
    right_title: str,
    image_height_ratio: float = 1.15,
) -> None:
    frame_ax = fig.add_subplot(subplot_spec)
    frame_ax.axis("off")
    frame_ax.text(
        0.0,
        1.035,
        f"({panel_label}) {panel_title}",
        transform=frame_ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=DARK,
        va="bottom",
    )

    inner = GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=subplot_spec,
        height_ratios=[image_height_ratio, 1.0],
        hspace=0.28,
        wspace=0.20,
    )
    image_ax = fig.add_subplot(inner[0, :])
    image = Image.open(original_path).convert("RGB")
    image_ax.imshow(image)
    image_ax.axis("off")

    gt_ax = fig.add_subplot(inner[1, 0])
    left_ax = fig.add_subplot(inner[1, 1])
    right_ax = fig.add_subplot(inner[1, 2])
    draw_structure(gt_ax, gt_record["gt_tokens"], "Ground truth", None)
    draw_structure(
        left_ax,
        left_record["pred_tokens"],
        left_title,
        left_record["teds"],
        gt_record["gt_tokens"],
    )
    draw_structure(
        right_ax,
        right_record["pred_tokens"],
        right_title,
        right_record["teds"],
        gt_record["gt_tokens"],
    )

    for ax in (gt_ax, left_ax, right_ax):
        ax.title.set_fontsize(8.2)


def make_standalone_compact_figure(
    output_dir: Path,
    stem: str,
    panel_label: str,
    panel_title: str,
    original_path: Path,
    filename: str,
    gt_record: dict,
    left_record: dict,
    left_title: str,
    right_record: dict,
    right_title: str,
) -> None:
    with Image.open(original_path) as source_image:
        image_aspect = source_image.width / source_image.height
    usable_width = 5.95
    structure_height = 2.8
    desired_image_height = usable_width / image_aspect
    figure_height = min(8.7, max(6.3, desired_image_height + 3.7))
    image_height_ratio = desired_image_height / structure_height

    fig = plt.figure(figsize=(6.4, figure_height), facecolor="white")
    spec = fig.add_gridspec(
        1,
        1,
        left=0.04,
        right=0.98,
        bottom=0.055,
        top=0.92,
    )[0, 0]
    draw_compact_case_panel(
        fig,
        spec,
        panel_label,
        panel_title,
        original_path,
        filename,
        gt_record,
        left_record,
        left_title,
        right_record,
        right_title,
        image_height_ratio=image_height_ratio,
    )

    fig.savefig(output_dir / f"{stem}.png", dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_combined_figure(
    output_dir: Path,
    project: Path,
    baseline: dict[int, dict],
    phase5e: dict[int, dict],
    full: dict[int, dict],
    no_bidir: dict[int, dict],
    no_geometry: dict[int, dict],
) -> None:
    fig = plt.figure(figsize=(10.2, 10.8), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        left=0.045,
        right=0.98,
        bottom=0.075,
        top=0.95,
        wspace=0.10,
        hspace=0.19,
    )

    cases = [
        (
            outer[0, 0],
            "a",
            "Merged-cell recovery",
            531,
            phase5e,
            baseline,
            "SLANet",
            phase5e,
            "DBM-SLANet",
        ),
        (
            outer[0, 1],
            "b",
            "Multi-level header recovery",
            650,
            phase5e,
            baseline,
            "SLANet",
            phase5e,
            "DBM-SLANet",
        ),
        (
            outer[1, 0],
            "c",
            "Bidirectional scan ablation",
            48,
            full,
            no_bidir,
            "No bidirectional",
            full,
            "Full model",
        ),
        (
            outer[1, 1],
            "d",
            "Geometry input ablation",
            4052,
            full,
            no_geometry,
            "No geometry",
            full,
            "Full model",
        ),
    ]

    for spec, label, title, idx, gt_data, left_data, left_title, right_data, right_title in cases:
        record = gt_data[idx]
        draw_compact_case_panel(
            fig,
            spec,
            label,
            title,
            project / record["image_path"],
            record["filename"],
            record,
            left_data[idx],
            left_title,
            right_data[idx],
            right_title,
        )

    legend = [
        Patch(facecolor=LIGHT_GREEN, edgecolor=GREEN, label="Correct predicted cell"),
        Patch(facecolor=LIGHT_RED, edgecolor=RED, label="Incorrect predicted cell"),
        Patch(facecolor="white", edgecolor=ORANGE, linestyle="--", label="Missing GT cell"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        fontsize=9,
    )

    png_path = output_dir / "real_effect_figures_combined_integrated.png"
    pdf_path = output_dir / "real_effect_figures_combined_integrated.pdf"
    fig.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()

    baseline = load_jsonl(project / "output/phase10_large_val_20260624/baseline_val5000_sample_teds.jsonl")
    phase5e = load_jsonl(project / "output/phase10_large_val_20260624/phase5e_val5000_sample_teds.jsonl")
    models_root = project / "output/phase10_more_models_val5000_20260710"
    full = load_jsonl(models_root / "e11c_mamba_lambda02_sample_teds.jsonl")
    no_bidir = load_jsonl(models_root / "e11e_bidir_false_sample_teds.jsonl")
    no_geometry = load_jsonl(models_root / "e11f_geom_zero_sample_teds.jsonl")

    comparison_cases = [
        (531, "real_comparison_1_span_recovery", "Comparison case 1: merged-cell structure recovered"),
        (650, "real_comparison_2_header_recovery", "Comparison case 2: multi-level header recovered"),
    ]
    for sample_idx, stem, title in comparison_cases:
        record = phase5e[sample_idx]
        make_figure(
            args.output_dir,
            stem,
            title,
            project / record["image_path"],
            record["filename"],
            record,
            baseline[sample_idx],
            "SLANet baseline",
            record,
            "DBM-SLANet",
        )

    ablation_cases = [
        (
            48,
            "real_ablation_1_bidirectional_scan",
            "Ablation case 1: bidirectional scan restores row-column relations",
            no_bidir,
            "Without bidirectional scan",
        ),
        (
            4052,
            "real_ablation_2_geometry_input",
            "Ablation case 2: geometry input restores merged-cell topology",
            no_geometry,
            "Without geometry input",
        ),
    ]
    for sample_idx, stem, title, ablation_data, ablation_title in ablation_cases:
        record = full[sample_idx]
        make_figure(
            args.output_dir,
            stem,
            title,
            project / record["image_path"],
            record["filename"],
            record,
            ablation_data[sample_idx],
            ablation_title,
            record,
            "Full DBM-SLANet",
        )

    make_combined_figure(
        args.output_dir,
        project,
        baseline,
        phase5e,
        full,
        no_bidir,
        no_geometry,
    )

    standalone_cases = [
        (
            "panel_a_merged_cell_recovery",
            "a",
            "Merged-cell recovery",
            531,
            phase5e,
            baseline,
            "SLANet",
            phase5e,
            "DBM-SLANet",
        ),
        (
            "panel_b_multilevel_header_recovery",
            "b",
            "Multi-level header recovery",
            650,
            phase5e,
            baseline,
            "SLANet",
            phase5e,
            "DBM-SLANet",
        ),
        (
            "panel_c_bidirectional_scan_ablation",
            "c",
            "Bidirectional scan ablation",
            48,
            full,
            no_bidir,
            "No bidirectional",
            full,
            "Full model",
        ),
        (
            "panel_d_geometry_input_ablation",
            "d",
            "Geometry input ablation",
            4052,
            full,
            no_geometry,
            "No geometry",
            full,
            "Full model",
        ),
    ]
    for stem, label, title, idx, gt_data, left_data, left_title, right_data, right_title in standalone_cases:
        record = gt_data[idx]
        make_standalone_compact_figure(
            args.output_dir,
            stem,
            label,
            title,
            project / record["image_path"],
            record["filename"],
            record,
            left_data[idx],
            left_title,
            right_data[idx],
            right_title,
        )

    print(f"Generated real qualitative figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
