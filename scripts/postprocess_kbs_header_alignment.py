#!/usr/bin/env python3
"""Align header titles and harmonize the input-table palette in the KBS figure."""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "论文材料" / "six_model_kbs_optimized_20260718"
SOURCE = OUTPUT_DIR / "comparison_case_1_kbs_optimized.png"
OUTPUT = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v2.png"
PDF_OUTPUT = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v2.pdf"

INK = "#25364A"
GT_FILL = "#E8EFF7"


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    width, height = image.size

    # Coordinates are proportional to the Matplotlib export and therefore remain
    # stable if the DPI changes while the page geometry stays the same.
    input_box = (
        round(width * 0.0865),
        round(height * 0.0275),
        round(width * 0.5130),
        round(height * 0.3580),
    )
    input_crop = image.crop(input_box)
    grayscale = np.asarray(ImageOps.grayscale(input_crop), dtype=np.float32)
    # Separate dark glyphs/rules from the nearly uniform cyan paper. A smooth
    # threshold preserves antialiasing while making the low-resolution source crisp.
    ink_strength = 1.0 / (1.0 + np.exp((grayscale - 158.0) / 9.0))
    ink_rgb = np.asarray(Image.new("RGB", (1, 1), INK), dtype=np.float32)[0, 0]
    fill_rgb = np.asarray(Image.new("RGB", (1, 1), GT_FILL), dtype=np.float32)[0, 0]
    harmonized_array = (
        fill_rgb[None, None, :] * (1.0 - ink_strength[:, :, None])
        + ink_rgb[None, None, :] * ink_strength[:, :, None]
    )
    harmonized = Image.fromarray(
        np.clip(harmonized_array, 0, 255).astype(np.uint8), mode="RGB"
    )
    image.paste(harmonized, input_box[:2])

    draw = ImageDraw.Draw(image)
    # Remove the lower Ground Truth title without touching its structure grid.
    draw.rectangle(
        (
            round(width * 0.735),
            round(height * 0.018),
            round(width * 0.965),
            round(height * 0.059),
        ),
        fill="white",
    )

    font_path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    title_font = ImageFont.truetype(font_path, round(height * 0.0172))
    ground_truth_center_x = round(width * 0.825)
    title_y = round(height * 0.0025)
    draw.text(
        (ground_truth_center_x, title_y),
        "Ground truth",
        font=title_font,
        fill=INK,
        anchor="ma",
    )

    image.save(OUTPUT, dpi=(500, 500), optimize=True)
    image.save(PDF_OUTPUT, "PDF", resolution=500)
    print(f"Saved {OUTPUT}")
    print(f"Saved {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
