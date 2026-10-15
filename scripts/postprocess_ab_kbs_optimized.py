#!/usr/bin/env python3
"""Restyle and align the existing A/B five-model qualitative figures locally."""

from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "论文材料" / "five_model_ab_comparisons_20260717"
OUTPUT_DIR = ROOT / "论文材料" / "five_model_ab_kbs_optimized_20260718"

CASES = [
    ("a", SOURCE_DIR / "qualitative_comparison_a.png", "qualitative_comparison_a_kbs"),
    (
        "b",
        SOURCE_DIR / "qualitative_comparison_b_complex.png",
        "qualitative_comparison_b_kbs",
    ),
]

INK = "#25364A"
GT_FILL = "#E8EFF7"
OURS_COLOR = "#2A7F74"

OLD_PALETTE = np.asarray(
    [
        (61, 111, 168),
        (234, 241, 248),
        (77, 141, 104),
        (232, 244, 236),
        (185, 84, 80),
        (250, 234, 232),
        (214, 138, 46),
    ],
    dtype=np.float32,
)

NEW_PALETTE = np.asarray(
    [
        (68, 114, 164),
        (232, 239, 247),
        (42, 127, 116),
        (226, 240, 236),
        (198, 93, 58),
        (247, 231, 224),
        (176, 122, 161),
    ],
    dtype=np.float32,
)

PANEL_TITLES = [
    "Ground Truth",
    "SLANet",
    "Cell Token",
    "Lite Mamba",
    "Horizontal Mamba",
    "DBM-SLANet (Ours)",
]


def palette_distance(array: np.ndarray) -> np.ndarray:
    pixels = array.astype(np.float32)
    distance = np.linalg.norm(
        pixels[:, :, None, :] - OLD_PALETTE[None, None, :, :], axis=3
    )
    return distance


def colored_mask(array: np.ndarray) -> np.ndarray:
    return (palette_distance(array).min(axis=2) < 52.0) & (array.mean(axis=2) < 250)


def recolor_structure(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    distance = palette_distance(array)
    nearest = distance.argmin(axis=2)
    minimum = distance.min(axis=2)
    for index, replacement in enumerate(NEW_PALETTE.astype(np.uint8)):
        mask = (nearest == index) & (minimum < 72.0)
        array[mask] = replacement
    return Image.fromarray(array, mode="RGB")


def harmonize_input(image: Image.Image) -> Image.Image:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    ink_strength = 1.0 / (1.0 + np.exp((grayscale - 158.0) / 9.0))
    ink_rgb = np.asarray(Image.new("RGB", (1, 1), INK), dtype=np.float32)[0, 0]
    fill_rgb = np.asarray(Image.new("RGB", (1, 1), GT_FILL), dtype=np.float32)[0, 0]
    result = (
        fill_rgb[None, None, :] * (1.0 - ink_strength[:, :, None])
        + ink_rgb[None, None, :] * ink_strength[:, :, None]
    )
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Unable to detect figure content")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def extract_panels(
    source: Image.Image, panel_label: str
) -> tuple[Image.Image, list[Image.Image]]:
    array = np.asarray(source.convert("RGB"))
    height, width = array.shape[:2]
    mask = colored_mask(array)
    top_fraction, split_fraction = (
        (0.30, 0.65) if panel_label == "a" else (0.34, 0.69)
    )
    row_ranges = [
        (round(height * top_fraction), round(height * split_fraction)),
        (round(height * split_fraction), height),
    ]

    panels: list[Image.Image] = []
    for row_index, (row_start, row_end) in enumerate(row_ranges):
        for column in range(3):
            x_start = round(width * column / 3)
            x_end = round(width * (column + 1) / 3)
            local_mask = mask[row_start:row_end, x_start:x_end]
            x1, y1, x2, y2 = tight_bbox(local_mask)
            pad = 4
            box = (
                max(0, x_start + x1 - pad),
                max(0, row_start + y1 - pad),
                min(width, x_start + x2 + pad),
                min(height, row_start + y2 + pad),
            )
            raw_panel = source.crop(box)
            raw_array = np.asarray(raw_panel.convert("RGB"))
            dense_rows = np.where(
                colored_mask(raw_array).sum(axis=1) > raw_panel.width * 0.45
            )[0]
            if len(dense_rows):
                trim_top = max(0, int(dense_rows.min()) - 3)
                raw_panel = raw_panel.crop(
                    (0, trim_top, raw_panel.width, raw_panel.height)
                )
            if row_index * 3 + column == 5:
                # The original Ours title is green and can be mistaken for panel
                # content by the palette detector; remove its residual title band.
                extra_trim = round(raw_panel.height * 0.085)
                raw_panel = raw_panel.crop(
                    (0, extra_trim, raw_panel.width, raw_panel.height)
                )
            panels.append(recolor_structure(raw_panel))

    first_panel_y = row_ranges[0][0]
    input_band_end = max(1, first_panel_y)
    input_start = round(height * 0.012)
    input_array = array[input_start:input_band_end]
    dark_mask = input_array.mean(axis=2) < 232
    # Remove the small global (a)/(b) label at the very top before finding the table.
    dark_mask[: max(1, round(height * 0.012))] = False
    x1, y1, x2, y2 = tight_bbox(dark_mask)
    input_crop = source.crop(
        (
            max(0, x1 - 4),
            max(0, input_start + y1 - 4),
            min(width, x2 + 4),
            min(height, input_start + y2 + 4),
        )
    )
    harmonized_input = harmonize_input(input_crop)
    horizontal_pad = max(6, round(harmonized_input.width * 0.008))
    harmonized_input = ImageOps.expand(
        harmonized_input,
        border=(horizontal_pad, 4, horizontal_pad, 4),
        fill=GT_FILL,
    )
    return harmonized_input, panels


def render_case(panel_label: str, source_path: Path, stem: str) -> None:
    source = Image.open(source_path).convert("RGB")
    input_image, panels = extract_panels(source, panel_label)
    input_aspect = input_image.width / input_image.height
    structure_aspect = min(2.15, max(1.35, input_aspect))

    page_width = 3600
    margin = 70
    column_gap = 70
    panel_width = (page_width - 2 * margin - 2 * column_gap) // 3
    panel_height = round(panel_width / structure_aspect)
    input_width = page_width - 2 * margin
    input_height = round(input_width / input_aspect)
    title_height = 82
    header_height = 78
    row_gap = 88
    panel_row_height = title_height + panel_height
    page_height = (
        header_height
        + input_height
        + 115
        + panel_row_height * 2
        + row_gap
        + 70
    )

    canvas = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    title_font = ImageFont.truetype(font_path, 50)
    panel_font = ImageFont.truetype(font_path, 43)

    draw.text((margin, 8), f"({panel_label})", font=title_font, fill=INK, anchor="la")
    draw.text(
        (page_width // 2, 8),
        "Input table image",
        font=title_font,
        fill=INK,
        anchor="ma",
    )
    resized_input = input_image.resize((input_width, input_height), Image.Resampling.LANCZOS)
    canvas.paste(resized_input, (margin, header_height))

    grid_top = header_height + input_height + 115
    for index, (title, panel) in enumerate(zip(PANEL_TITLES, panels)):
        row = index // 3
        column = index % 3
        x = margin + column * (panel_width + column_gap)
        y = grid_top + row * (panel_row_height + row_gap)
        color = OURS_COLOR if title == "DBM-SLANet (Ours)" else INK
        draw.text((x, y), title, font=panel_font, fill=color, anchor="la")
        normalized = panel.resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS
        )
        canvas.paste(normalized, (x, y + title_height))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    canvas.save(png_path, dpi=(500, 500), optimize=True)
    canvas.save(pdf_path, "PDF", resolution=500)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    for panel_label, source_path, stem in CASES:
        render_case(panel_label, source_path, stem)


if __name__ == "__main__":
    main()
