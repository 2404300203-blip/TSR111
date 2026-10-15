#!/usr/bin/env python3
"""Remove the error-fill tint from the marked row in panel (c)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "论文材料" / "six_model_kbs_optimized_20260718"
SOURCE = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v3_aligned.png"
OUTPUT = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v4_panel_c_white.png"
PDF_OUTPUT = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v4_panel_c_white.pdf"

ERROR_FILL = np.array((247, 231, 224), dtype=np.int16)
WHITE = np.array((255, 255, 255), dtype=np.uint8)


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    pixels = np.asarray(image).copy()
    height, width = pixels.shape[:2]

    # Last error-tinted prediction row in panel (c), immediately above the
    # green correctly predicted rows. Ratios keep the edit stable across DPI.
    x1, x2 = round(width * 0.688), round(width * 0.997)
    y1, y2 = round(height * 0.602), round(height * 0.621)
    region = pixels[y1:y2, x1:x2]

    color_distance = np.max(
        np.abs(region.astype(np.int16) - ERROR_FILL), axis=2
    )
    fill_mask = color_distance <= 10
    replaced_pixels = int(fill_mask.sum())
    if replaced_pixels < 10_000:
        raise RuntimeError(
            f"Marked panel fill was not found; only {replaced_pixels} pixels matched"
        )

    region[fill_mask] = WHITE
    result = Image.fromarray(pixels, mode="RGB")
    result.save(OUTPUT, dpi=(500, 500), optimize=True)
    result.save(PDF_OUTPUT, "PDF", resolution=500)
    print(f"Whitened {replaced_pixels} fill pixels")
    print(f"Saved {OUTPUT}")
    print(f"Saved {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
