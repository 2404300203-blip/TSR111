#!/usr/bin/env python3
"""Build a small PaddleOCR-ready FinTabNet split from PDF pages.

FinTabNet labels point to single-page PDFs and store table/cell boxes in PDF
points. PaddleOCR table models need cropped table images plus cell boxes in the
cropped image coordinate system. This utility downloads the needed PDFs from
Kaggle, renders each page with poppler, crops the table region, and writes a
PubTabNet-style JSONL label file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


DEFAULT_PDFTOPPM = (
    "/Users/table/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pdftoppm"
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def download_pdf(record_filename: str, pdf_root: Path, kaggle_bin: str) -> Path:
    rel = Path(record_filename)
    pdf_path = pdf_root / rel
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            kaggle_bin,
            "datasets",
            "download",
            "-d",
            "jiongjiong/fintabnet",
            "-f",
            f"fintabnet/pdf/{record_filename}",
            "--force",
            "-p",
            str(pdf_path.parent),
        ]
    )
    downloaded = pdf_path.parent / rel.name
    if not downloaded.exists():
        raise FileNotFoundError(f"Kaggle did not create expected PDF: {downloaded}")
    return downloaded


def render_pdf(pdf_path: Path, pdftoppm_bin: str, dpi: int) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "page"
        run([pdftoppm_bin, "-r", str(dpi), "-png", "-singlefile", str(pdf_path), str(out_prefix)])
        image_path = out_prefix.with_suffix(".png")
        with Image.open(image_path) as im:
            return im.convert("RGB")


def pdf_box_to_pixels(box, page_w: float, page_h: float, scale: float):
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    left, right = sorted((x1, x2))
    low_y, high_y = sorted((y1, y2))
    return [
        left * scale,
        (page_h - high_y) * scale,
        right * scale,
        (page_h - low_y) * scale,
    ]


def clamp_box(box, width: int, height: int):
    left, top, right, bottom = box
    left = max(0, min(width - 1, int(round(left))))
    top = max(0, min(height - 1, int(round(top))))
    right = max(left + 1, min(width, int(round(right))))
    bottom = max(top + 1, min(height, int(round(bottom))))
    return [left, top, right, bottom]


def normalize_cell_text(cell: dict) -> dict:
    out = dict(cell)
    tokens = out.get("tokens")
    if isinstance(tokens, list):
        out["tokens"] = tokens
    elif "pdf_text_content" in out:
        out["tokens"] = list(str(out.get("pdf_text_content") or ""))
    elif "json_text_content" in out:
        out["tokens"] = list(str(out.get("json_text_content") or ""))
    else:
        out["tokens"] = []
    return out


def convert_record(record: dict, image: Image.Image, image_name: str, dpi: int):
    page_w = 72.0 * image.width / dpi
    page_h = 72.0 * image.height / dpi
    scale = dpi / 72.0

    table_box_px = clamp_box(pdf_box_to_pixels(record["bbox"], page_w, page_h, scale), image.width, image.height)
    crop = image.crop(tuple(table_box_px))
    crop_w, crop_h = crop.size
    table_left, table_top, _, _ = table_box_px

    cells = []
    for cell in record.get("html", {}).get("cells", []):
        if not isinstance(cell, dict):
            continue
        out_cell = normalize_cell_text(cell)
        bbox = out_cell.get("bbox") or out_cell.get("pdf_bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            cell_px = pdf_box_to_pixels(bbox, page_w, page_h, scale)
            rel = [
                cell_px[0] - table_left,
                cell_px[1] - table_top,
                cell_px[2] - table_left,
                cell_px[3] - table_top,
            ]
            out_cell["bbox"] = clamp_box(rel, crop_w, crop_h)
        elif "bbox" in out_cell:
            out_cell.pop("bbox", None)
        cells.append(out_cell)

    out_record = {
        "filename": f"images/{image_name}",
        "html": {
            "structure": {"tokens": record["html"]["structure"]["tokens"]},
            "cells": cells,
        },
    }
    return crop, out_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--pdf-root", default="")
    parser.add_argument("--kaggle-bin", default=str(Path.home() / ".local/bin/kaggle"))
    parser.add_argument("--pdftoppm-bin", default=DEFAULT_PDFTOPPM)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images"
    pdf_root = Path(args.pdf_root) if args.pdf_root else out_dir / "pdf"
    label_path = out_dir / f"fintabnet_original_{args.split_name}_smoke{args.limit}.jsonl"
    manifest_path = out_dir / f"fintabnet_original_{args.split_name}_smoke{args.limit}_manifest.json"
    image_dir.mkdir(parents=True, exist_ok=True)
    pdf_root.mkdir(parents=True, exist_ok=True)

    converted = []
    skipped = []
    for index, record in enumerate(iter_jsonl(input_jsonl)):
        if len(converted) >= args.limit:
            break
        try:
            pdf_path = download_pdf(record["filename"], pdf_root, args.kaggle_bin)
            page = render_pdf(pdf_path, args.pdftoppm_bin, args.dpi)
            pdf_stem = Path(record["filename"]).with_suffix("").as_posix().replace("/", "_")
            image_name = f"{pdf_stem}_table_{record.get('table_id', index)}.png"
            crop, out_record = convert_record(record, page, image_name, args.dpi)
            crop.save(image_dir / image_name)
            converted.append(out_record)
            print(f"[{len(converted):04d}] {record['filename']} -> images/{image_name}")
        except Exception as exc:  # keep smoke generation moving and report misses.
            skipped.append({"filename": record.get("filename"), "error": repr(exc)})
            print(f"[skip] {record.get('filename')}: {exc}")

    with label_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(out_dir),
        "label_file": str(label_path),
        "image_dir": str(image_dir),
        "pdf_root": str(pdf_root),
        "dpi": args.dpi,
        "converted": len(converted),
        "skipped": skipped,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
