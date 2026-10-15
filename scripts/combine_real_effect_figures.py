#!/usr/bin/env python3
"""Combine four real qualitative effect figures into one 2x2 composite."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FILES = [
    "real_comparison_1_span_recovery.png",
    "real_comparison_2_header_recovery.png",
    "real_ablation_1_bidirectional_scan.png",
    "real_ablation_2_geometry_input.png",
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("figure_dir", type=Path)
    args = parser.parse_args()

    figure_dir = args.figure_dir.resolve()
    images = [Image.open(figure_dir / name).convert("RGB") for name in FILES]
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)

    outer_margin = 80
    column_gap = 70
    row_gap = 80
    header_height = 150
    canvas_width = outer_margin * 2 + cell_width * 2 + column_gap
    canvas_height = outer_margin * 2 + header_height * 2 + cell_height * 2 + row_gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(78)
    panel_font = load_font(72)
    text_color = "#263442"

    row_headers = ["Comparison experiments", "Ablation experiments"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for row in range(2):
        header_y = outer_margin + row * (header_height + cell_height + row_gap)
        header = row_headers[row]
        bbox = draw.textbbox((0, 0), header, font=header_font)
        header_x = (canvas_width - (bbox[2] - bbox[0])) // 2
        draw.text((header_x, header_y), header, fill=text_color, font=header_font)

        image_y = header_y + header_height
        for col in range(2):
            idx = row * 2 + col
            image = images[idx]
            cell_x = outer_margin + col * (cell_width + column_gap)
            image_x = cell_x + (cell_width - image.width) // 2
            centered_y = image_y + (cell_height - image.height) // 2
            canvas.paste(image, (image_x, centered_y))
            draw.text(
                (cell_x + 18, image_y + 10),
                panel_labels[idx],
                fill=text_color,
                font=panel_font,
            )

    png_path = figure_dir / "real_effect_figures_combined_2x2.png"
    pdf_path = figure_dir / "real_effect_figures_combined_2x2.pdf"
    canvas.save(png_path, dpi=(300, 300), optimize=True)
    canvas.save(pdf_path, "PDF", resolution=300.0)
    print(f"Created {png_path}")
    print(f"Created {pdf_path}")


if __name__ == "__main__":
    main()
