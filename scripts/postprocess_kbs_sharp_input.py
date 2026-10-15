#!/usr/bin/env python3
"""Replace the input-table panel with a sharper rendering of the raw image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "论文材料" / "six_model_kbs_optimized_20260718"
SOURCE_FIGURE = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v4_panel_c_white.png"
RAW_INPUT = OUTPUT_DIR / "source_inputs" / "PMC4819627_004_00.png"
OUTPUT = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v5_sharp_input.png"
PDF_OUTPUT = OUTPUT_DIR / "comparison_case_1_kbs_optimized_v5_sharp_input.pdf"

INK = np.array((37, 54, 74), dtype=np.float32)
FILL = np.array((232, 239, 247), dtype=np.float32)


def render_sharp_input(raw: Image.Image, size: tuple[int, int]) -> Image.Image:
    upscaled = raw.resize(size, Image.Resampling.LANCZOS)
    upscaled = upscaled.filter(
        ImageFilter.UnsharpMask(radius=1.1, percent=120, threshold=2)
    )
    grayscale = np.asarray(upscaled.convert("L"), dtype=np.float32)
    ink_strength = 1.0 / (1.0 + np.exp((grayscale - 158.0) / 4.5))
    rendered = (
        FILL[None, None, :] * (1.0 - ink_strength[:, :, None])
        + INK[None, None, :] * ink_strength[:, :, None]
    )
    return Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8), mode="RGB")


def main() -> None:
    figure = Image.open(SOURCE_FIGURE).convert("RGB")
    raw_input = Image.open(RAW_INPUT).convert("RGB")
    width, height = figure.size

    input_box = (
        round(width * 0.0865),
        round(height * 0.0275),
        round(width * 0.5130),
        round(height * 0.3580),
    )
    target_size = (input_box[2] - input_box[0], input_box[3] - input_box[1])
    sharp_input = render_sharp_input(raw_input, target_size)
    figure.paste(sharp_input, input_box[:2])

    figure.save(OUTPUT, dpi=(500, 500), optimize=True)
    figure.save(PDF_OUTPUT, "PDF", resolution=500)
    print(f"Raw input: {raw_input.size[0]}x{raw_input.size[1]}")
    print(f"Rendered input: {target_size[0]}x{target_size[1]}")
    print(f"Saved {OUTPUT}")
    print(f"Saved {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
