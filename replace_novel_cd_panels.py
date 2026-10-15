from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path("/Users/table/TSR/论文材料/header_metrics_effect_legend_20260914")
SRC = ROOT / "header_metrics_real_effect_legend_clean.png"
OUT = ROOT / "header_metrics_real_effect_legend_clean_novel_cd.png"
MD = ROOT / "header_metrics_real_effect_legend_clean_novel_cd.md"

QROOT = Path(
    "/Users/table/TSR/论文材料/kbs_appendix_experiments_20260803/effect_figures"
)


def get_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def crop_table(path: Path, box):
    """Crop only the original table, excluding the effect labels/diagrams."""
    return Image.open(path).convert("RGB").crop(box)


def fit_inside(im: Image.Image, size):
    return ImageOps.contain(im, size, method=Image.Resampling.LANCZOS)


def main():
    base = Image.open(SRC).convert("RGB")
    width, height = base.size
    top_h = 1830
    bottom_y = 1930
    bottom_h = height - bottom_y
    half_w = width // 2

    # The two source tables are from appendix cases and are not the paper's
    # bidirectional-scan or geometry-input ablation examples.
    c_table = crop_table(QROOT / "Q_success_02.png", (900, 335, 3050, 1390))
    d_table = crop_table(QROOT / "Q_complement_02.png", (1150, 335, 2850, 1390))

    out = base.copy()
    # Erase the old lower panels completely while preserving the unchanged
    # upper row pixel-for-pixel.
    out.paste("white", (0, bottom_y, width, height))
    draw = ImageDraw.Draw(out)
    title_font = get_font(86, bold=True)
    navy = "#263746"

    titles = [
        "(c) Grouped-row header example",
        "(d) Hierarchical category header example",
    ]
    tables = [c_table, d_table]
    for i, (title, table) in enumerate(zip(titles, tables)):
        x0 = i * half_w
        # Align titles with the existing KBS-style left margin.
        draw.text((x0 + 175, bottom_y + 62), title, fill=navy, font=title_font)
        # Leave a compact, consistent gap below the title and fit the table
        # without distorting its aspect ratio or clipping its last row.
        preview = fit_inside(table, (half_w - 360, bottom_h - 205))
        px = x0 + (half_w - preview.width) // 2
        py = bottom_y + 175 + (bottom_h - 205 - preview.height) // 2
        out.paste(preview, (px, py))

    out.save(OUT, dpi=(600, 600))
    MD.write_text(
        """# Header-metric qualitative examples (revised)

This revision keeps panels **(a)** and **(b)** unchanged and replaces panels
**(c)** and **(d)** with two appendix table cases that are not used in the
paper's bidirectional-scan or geometry-input ablation figures.

- **(c) Grouped-row header example**: a multi-row grouped table with repeated
  gene blocks and paired subentries. It stresses row grouping and vertical
  header continuity.
- **(d) Hierarchical category header example**: a clinical summary table with
  nested category rows and multiple value columns. It stresses hierarchical
  header depth and row/column span boundaries.

Only the original table regions are shown; the old ablation labels, TEDS
annotations, and schematic cell diagrams are not included. The output is
written as a new PNG so the previous figure remains untouched.

Source cases:

- `Q_success_02.png` (appendix case, PMC4709790_002_00)
- `Q_complement_02.png` (appendix case, PMC5985617_002_00)
""",
        encoding="utf-8",
    )
    print(OUT)
    print(MD)


if __name__ == "__main__":
    main()
