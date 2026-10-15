from pathlib import Path
from PIL import Image


SRC = Path(
    "/Users/table/TSR/论文材料/header_metrics_effect_legend_20260914/"
    "header_metrics_real_effect_legend_final.png"
)
OUT = SRC.with_name("header_metrics_real_effect_legend_clean.png")


def main() -> None:
    image = Image.open(SRC).convert("RGB")
    width, height = image.size
    half_w, half_h = width // 2, height // 2

    # The upper two source panels contain taller real tables than the lower
    # two. Crop immediately below each table, removing the lower schematic
    # illustrations and their explanatory captions without resampling.
    top_h = 1830
    # End before the lower model-output captions begin; this avoids leaving
    # clipped fragments such as “Full model” at the bottom edge.
    bottom_h = 1380
    panels = [
        image.crop((0, 0, half_w, top_h)),
        image.crop((half_w, 0, width, top_h)),
        image.crop((0, half_h, half_w, half_h + bottom_h)),
        image.crop((half_w, half_h, width, half_h + bottom_h)),
    ]
    row_gap = 100
    canvas = Image.new("RGB", (width, top_h + row_gap + bottom_h), "white")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (half_w, 0))
    canvas.paste(panels[2], (0, top_h + row_gap))
    canvas.paste(panels[3], (half_w, top_h + row_gap))
    canvas.save(OUT, dpi=(600, 600))
    print(OUT)


if __name__ == "__main__":
    main()
