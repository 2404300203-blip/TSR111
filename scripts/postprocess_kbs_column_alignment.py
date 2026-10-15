#!/usr/bin/env python3
"""Enforce exact a/d, b/e, and c/f column alignment in the local KBS image."""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "论文材料" / "six_model_kbs_optimized_20260718"
SOURCE = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v2.png"
OUTPUT = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v3_aligned.png"
PDF_OUTPUT = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v3_aligned.pdf"

INK = "#25364A"
OURS_COLOR = "#2A7F74"


def move_region(
    image: Image.Image,
    box: tuple[int, int, int, int],
    delta_x: int,
) -> None:
    crop = image.crop(box)
    ImageDraw.Draw(image).rectangle(box, fill="white")
    image.paste(crop, (box[0] + delta_x, box[1]))


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    width, height = image.size

    # Correct the small auto-layout offset of the upper middle structure so that
    # b and e share exactly the same centerline. The other two pairs already match.
    move_region(
        image,
        (
            round(width * 0.3337),
            round(height * 0.422),
            round(width * 0.6667),
            round(height * 0.686),
        ),
        delta_x=4,
    )

    draw = ImageDraw.Draw(image)
    title_bands = [
        (round(height * 0.391), round(height * 0.425)),
        (round(height * 0.706), round(height * 0.747)),
    ]
    for y1, y2 in title_bands:
        draw.rectangle((0, y1, width, y2), fill="white")

    font_path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    title_font = ImageFont.truetype(font_path, round(height * 0.0166))
    left_edges = [round(width * value) for value in (0.0081, 0.3478, 0.6877)]
    labels = [
        ("(a) SLANet", INK),
        ("(b) Cell Token", INK),
        ("(c) Lite Mamba", INK),
        ("(d) Horizontal Mamba", INK),
        ("(e) Vertical Mamba", INK),
        ("(f) DBM-SLANet (Ours)", OURS_COLOR),
    ]
    title_y = [round(height * 0.393), round(height * 0.711)]
    for index, (label, color) in enumerate(labels):
        row = index // 3
        column = index % 3
        draw.text(
            (left_edges[column], title_y[row]),
            label,
            font=title_font,
            fill=color,
            anchor="la",
        )

    image.save(OUTPUT, dpi=(500, 500), optimize=True)
    image.save(PDF_OUTPUT, "PDF", resolution=500)
    print(f"Saved {OUTPUT}")
    print(f"Saved {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
