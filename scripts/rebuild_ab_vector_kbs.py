#!/usr/bin/env python3
"""Rebuild sharp KBS-style A/B figures directly from the original vector PDFs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import fitz
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "论文材料" / "five_model_ab_comparisons_20260717"
OUTPUT_DIR = ROOT / "论文材料" / "five_model_ab_kbs_optimized_20260718"

CASES = [
    {
        "label": "a",
        "source": SOURCE_DIR / "qualitative_comparison_a.pdf",
        "stem": "qualitative_comparison_a_kbs_vector",
        "rows": [(245.0, 445.0), (488.0, 688.8)],
    },
    {
        "label": "b",
        "source": SOURCE_DIR / "qualitative_comparison_b_complex.pdf",
        "stem": "qualitative_comparison_b_kbs_vector",
        "rows": [(290.0, 500.0), (530.0, 734.1)],
    },
]

PANEL_TITLES = [
    "Ground Truth",
    "SLANet",
    "Cell Token",
    "Lite Mamba",
    "Horizontal Mamba",
    "DBM-SLANet (Ours)",
]

INK = (37 / 255, 54 / 255, 74 / 255)
OURS = (42 / 255, 127 / 255, 116 / 255)
GT_FILL = "#E8EFF7"
SPAN_COLOR = (123 / 255, 135 / 255, 148 / 255)

COLOR_REPLACEMENTS = {
    b"0.1490196078 0.2039215686 0.2588235294": b"0.1450980392 0.2117647059 0.2901960784",
    b"0.2392156863 0.4352941176 0.6588235294": b"0.2666666667 0.4470588235 0.6431372549",
    b"0.3019607843 0.5529411765 0.4078431373": b"0.1647058824 0.4980392157 0.4549019608",
    b"0.7254901961 0.3294117647 0.3137254902": b"0.7764705882 0.3647058824 0.2274509804",
    b"0.8392156863 0.5411764706 0.1803921569": b"0.6901960784 0.4784313725 0.6313725490",
    b"0.9098039216 0.9568627451 0.9254901961": b"0.8862745098 0.9411764706 0.9254901961",
    b"0.9176470588 0.9450980392 0.9725490196": b"0.9098039216 0.9372549020 0.9686274510",
    b"0.9803921569 0.9176470588 0.9098039216": b"0.9686274510 0.9058823529 0.8784313725",
}


def recolor_pdf(source: Path, destination: Path) -> None:
    reader = PdfReader(str(source))
    writer = PdfWriter()
    for page in reader.pages:
        data = page.get_contents().get_data()
        for old, new in COLOR_REPLACEMENTS.items():
            data = data.replace(old, new)
        stream = DecodedStreamObject()
        stream.set_data(data)
        page.replace_contents(stream)
        writer.add_page(page)
    with destination.open("wb") as handle:
        writer.write(handle)


def harmonize_input(image: Image.Image) -> Image.Image:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    ink_strength = 1.0 / (1.0 + np.exp((grayscale - 158.0) / 9.0))
    ink_rgb = np.asarray(Image.new("RGB", (1, 1), "#25364A"), dtype=np.float32)[0, 0]
    fill_rgb = np.asarray(Image.new("RGB", (1, 1), GT_FILL), dtype=np.float32)[0, 0]
    result = (
        fill_rgb[None, None, :] * (1.0 - ink_strength[:, :, None])
        + ink_rgb[None, None, :] * ink_strength[:, :, None]
    )
    crisp = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")
    return crisp.resize((crisp.width * 6, crisp.height * 6), Image.Resampling.NEAREST)


def extract_input_png(document: fitz.Document) -> tuple[bytes, float]:
    page = document[0]
    image_info = page.get_image_info(xrefs=True)[0]
    extracted = document.extract_image(image_info["xref"])
    image = Image.open(BytesIO(extracted["image"])).convert("RGB")
    harmonized = harmonize_input(image)
    buffer = BytesIO()
    harmonized.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), image.width / image.height


def find_panel_clips(
    page: fitz.Page, row_ranges: list[tuple[float, float]]
) -> list[fitz.Rect]:
    drawings = page.get_drawings()
    page_width = page.rect.width
    clips = []
    for y0, y1 in row_ranges:
        for column in range(3):
            x0 = page_width * column / 3
            x1 = page_width * (column + 1) / 3
            rects = []
            for drawing in drawings:
                rect = drawing["rect"]
                center = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
                if x0 <= center[0] <= x1 and y0 <= center[1] <= y1:
                    rects.append(rect)
            if not rects:
                raise ValueError(f"No vector drawings found in panel {len(clips)}")
            union = fitz.Rect(rects[0])
            for rect in rects[1:]:
                union.include_rect(rect)
            clips.append(union + (-1.5, -1.5, 1.5, 1.5))
    return clips


def add_span_guides(
    output_page: fitz.Page,
    source_page: fitz.Page,
    source_clip: fitz.Rect,
    target_rect: fitz.Rect,
) -> None:
    drawings = [drawing for drawing in source_page.get_drawings() if drawing.get("fill")]
    scale_x = target_rect.width / source_clip.width
    scale_y = target_rect.height / source_clip.height
    for word in source_page.get_text("words"):
        label = word[4]
        match = re.fullmatch(r"(\d+)x(\d+)", label)
        if not match:
            continue
        word_rect = fitz.Rect(word[:4])
        center = fitz.Point(
            (word_rect.x0 + word_rect.x1) / 2,
            (word_rect.y0 + word_rect.y1) / 2,
        )
        if not source_clip.contains(center):
            continue
        containers = [
            drawing["rect"]
            for drawing in drawings
            if drawing["rect"].contains(center)
        ]
        if not containers:
            continue
        cell = min(containers, key=lambda rect: rect.get_area())
        mapped = fitz.Rect(
            target_rect.x0 + (cell.x0 - source_clip.x0) * scale_x,
            target_rect.y0 + (cell.y0 - source_clip.y0) * scale_y,
            target_rect.x0 + (cell.x1 - source_clip.x0) * scale_x,
            target_rect.y0 + (cell.y1 - source_clip.y0) * scale_y,
        )
        rowspan, colspan = int(match.group(1)), int(match.group(2))
        for index in range(1, colspan):
            x = mapped.x0 + mapped.width * index / colspan
            output_page.draw_line(
                fitz.Point(x, mapped.y0),
                fitz.Point(x, mapped.y1),
                color=SPAN_COLOR,
                width=0.55,
                dashes="[2.5 1.8] 0",
                overlay=True,
            )
        for index in range(1, rowspan):
            y = mapped.y0 + mapped.height * index / rowspan
            output_page.draw_line(
                fitz.Point(mapped.x0, y),
                fitz.Point(mapped.x1, y),
                color=SPAN_COLOR,
                width=0.55,
                dashes="[2.5 1.8] 0",
                overlay=True,
            )


def build_case(case: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_pdf = OUTPUT_DIR / f".{case['stem']}_source.pdf"
    recolor_pdf(case["source"], temporary_pdf)

    original_document = fitz.open(case["source"])
    recolored_document = fitz.open(temporary_pdf)
    source_page = recolored_document[0]
    clips = find_panel_clips(source_page, case["rows"])
    input_png, input_aspect = extract_input_png(original_document)
    structure_aspect = min(2.15, max(1.35, input_aspect))

    page_width = 720.0
    margin = 20.0
    column_gap = 14.0
    panel_width = (page_width - 2 * margin - 2 * column_gap) / 3
    panel_height = panel_width / structure_aspect
    input_width = page_width - 2 * margin
    input_height = input_width / input_aspect
    header_height = 29.0
    panel_title_height = 20.0
    row_gap = 26.0
    grid_top = header_height + input_height + 27.0
    page_height = (
        grid_top
        + 2 * (panel_title_height + panel_height)
        + row_gap
        + 18.0
    )

    output_document = fitz.open()
    page = output_document.new_page(width=page_width, height=page_height)
    page.insert_text(
        fitz.Point(margin, 16),
        f"({case['label']})",
        fontname="hebo",
        fontsize=11.5,
        color=INK,
    )
    page.insert_textbox(
        fitz.Rect(0, 4, page_width, 24),
        "Input table image",
        fontname="hebo",
        fontsize=12.5,
        color=INK,
        align=fitz.TEXT_ALIGN_CENTER,
    )
    page.insert_image(
        fitz.Rect(margin, header_height, margin + input_width, header_height + input_height),
        stream=input_png,
        keep_proportion=False,
    )

    for index, (title, source_clip) in enumerate(zip(PANEL_TITLES, clips)):
        row = index // 3
        column = index % 3
        x = margin + column * (panel_width + column_gap)
        title_y = grid_top + row * (panel_title_height + panel_height + row_gap)
        target = fitz.Rect(
            x,
            title_y + panel_title_height,
            x + panel_width,
            title_y + panel_title_height + panel_height,
        )
        page.insert_text(
            fitz.Point(x, title_y + 12),
            title,
            fontname="hebo",
            fontsize=9.4,
            color=OURS if title == "DBM-SLANet (Ours)" else INK,
        )
        page.show_pdf_page(
            target,
            recolored_document,
            0,
            clip=source_clip,
            keep_proportion=False,
            overlay=True,
        )
        add_span_guides(page, source_page, source_clip, target)

    pdf_path = OUTPUT_DIR / f"{case['stem']}.pdf"
    png_path = OUTPUT_DIR / f"{case['stem']}.png"
    output_document.save(pdf_path, garbage=4, deflate=True)
    render_document = fitz.open(pdf_path)
    pixmap = render_document[0].get_pixmap(
        matrix=fitz.Matrix(500 / 72, 500 / 72), alpha=False
    )
    pixmap.save(png_path)
    render_document.close()
    output_document.close()
    recolored_document.close()
    original_document.close()
    temporary_pdf.unlink(missing_ok=True)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    for case in CASES:
        build_case(case)


if __name__ == "__main__":
    main()
