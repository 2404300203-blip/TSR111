#!/usr/bin/env python3
"""Remove TEDS text rows from the current six-model comparison PNG files."""

from pathlib import Path

from PIL import Image, ImageDraw


FIGURE_DIR = Path("论文材料/six_model_comparisons_20260714")

# Pixel bands for the two model-title rows in each final high-resolution image.
TEDS_BANDS = {
    "comparison_case_1_six_models.png": [(1775, 1836), (3241, 3302)],
    "comparison_case_2_six_models.png": [(2141, 2202), (3600, 3661)],
    "comparison_case_3_six_models.png": [(1519, 1580), (2991, 3052)],
    "comparison_case_4_six_models.png": [(1320, 1381), (2797, 2858)],
}


def main() -> None:
    for filename, bands in TEDS_BANDS.items():
        path = FIGURE_DIR / filename
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for top, bottom in bands:
            draw.rectangle((0, top, image.width, bottom), fill="white")
        image.save(path, dpi=(500, 500), optimize=True)
        image.save(path.with_suffix(".pdf"), "PDF", resolution=300.0)
        print(f"Updated {path}")


if __name__ == "__main__":
    main()
