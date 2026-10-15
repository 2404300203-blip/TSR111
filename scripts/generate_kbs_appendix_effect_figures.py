#!/usr/bin/env python3
"""Render additional non-main-text qualitative cases from real CTA-500 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from generate_real_qualitative_effect_figures import (
    DARK, GREEN, LIGHT_GREEN, LIGHT_RED, ORANGE, RED, draw_structure,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "remote_results/kbs_appendix_audit_20260803/cta500_sample_truth"
OUT = ROOT / "论文材料/kbs_appendix_experiments_20260803/effect_figures"
IMAGES = OUT / "source_inputs"

MODEL_FILES = {
    "SLANet": "baseline_cta500_sample_teds.jsonl",
    "Horizontal": "expC_horizontal_cta500_sample_teds.jsonl",
    "Vertical": "expD_vertical_cta500_sample_teds.jsonl",
    "DBM-SLANet": "phase5e_cta500_sample_teds.jsonl",
}

CASES = [
    ("success_01", "PMC5260049_003_00.png", "DBM exact recovery"),
    ("success_02", "PMC4709790_002_00.png", "DBM exact recovery"),
    ("complement_01", "PMC4819627_004_00.png", "Dual-branch complementarity"),
    ("complement_02", "PMC5985617_002_00.png", "Dual-branch complementarity"),
    ("confidence_failure_01", "PMC5618586_005_00.png", "High-confidence failure"),
    ("confidence_failure_02", "PMC5352430_002_00.png", "High-confidence failure"),
    ("regression_01", "PMC5328336_008_01.png", "DBM regression"),
    ("regression_02", "PMC4419394_001_00.png", "DBM regression"),
]


def load(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[row["filename"]] = row
    return result


def legends() -> list[Patch]:
    return [
        Patch(facecolor=LIGHT_GREEN, edgecolor=GREEN, label="Correct predicted cell"),
        Patch(facecolor=LIGHT_RED, edgecolor=RED, label="Incorrect predicted cell"),
        Patch(facecolor="white", edgecolor=ORANGE, linestyle="--", label="Missing GT cell"),
    ]


def render_case(stem: str, filename: str, title: str, data: dict[str, dict]) -> None:
    gt = data["DBM-SLANet"][filename]
    fig = plt.figure(figsize=(12.2, 5.7), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.32, wspace=0.22)
    image_ax = fig.add_subplot(grid[0, :])
    image_ax.imshow(Image.open(IMAGES / filename).convert("RGB"))
    image_ax.axis("off")
    image_ax.set_title(f"Input table: {filename}", fontsize=10.5, color=DARK, pad=5)

    axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    draw_structure(axes[0], gt["gt_tokens"], "Ground truth", None)
    draw_structure(
        axes[1], data["SLANet"][filename]["pred_tokens"], "SLANet",
        data["SLANet"][filename]["teds"], gt["gt_tokens"],
    )
    draw_structure(
        axes[2], gt["pred_tokens"], "DBM-SLANet", gt["teds"], gt["gt_tokens"],
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", color=DARK, y=0.99)
    fig.legend(handles=legends(), loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0.015, 0.075, 0.985, 0.95))
    for ext, kwargs in (("png", {"dpi": 500}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / f"Q_{stem}.{ext}", bbox_inches="tight", facecolor="white", pad_inches=0.04, **kwargs)
    plt.close(fig)


def render_pair(stem: str, pair: list[tuple[str, str, str]], data: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(14.8, 6.2), facecolor="white")
    for row_index, (_, filename, title) in enumerate(pair):
        gt = data["DBM-SLANet"][filename]
        axes[row_index, 0].imshow(Image.open(IMAGES / filename).convert("RGB"))
        axes[row_index, 0].axis("off")
        axes[row_index, 0].set_title(f"({chr(97 + row_index)}) Input", loc="left", fontsize=9.5, fontweight="bold")
        draw_structure(axes[row_index, 1], gt["gt_tokens"], "Ground truth", None)
        for col, model in enumerate(("SLANet", "Horizontal", "Vertical"), start=2):
            record = data[model][filename]
            draw_structure(axes[row_index, col], record["pred_tokens"], model, record["teds"], gt["gt_tokens"])
        draw_structure(axes[row_index, 5], gt["pred_tokens"], "DBM-SLANet", gt["teds"], gt["gt_tokens"])
        axes[row_index, 0].text(0, -0.08, title, transform=axes[row_index, 0].transAxes, fontsize=8, color=DARK)
    fig.legend(handles=legends(), loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.99), h_pad=1.25, w_pad=0.5)
    for ext, kwargs in (("png", {"dpi": 500}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / f"QP_{stem}.{ext}", bbox_inches="tight", facecolor="white", pad_inches=0.04, **kwargs)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {name: load(SOURCE / filename) for name, filename in MODEL_FILES.items()}
    for stem, filename, title in CASES:
        render_case(stem, filename, title, data)
    for index in range(0, len(CASES), 2):
        render_pair(CASES[index][0].rsplit("_", 1)[0], CASES[index:index + 2], data)
    print(f"Rendered {len(CASES)} cases and {len(CASES) // 2} paired figures")


if __name__ == "__main__":
    main()
