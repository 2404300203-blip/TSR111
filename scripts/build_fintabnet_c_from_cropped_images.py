#!/usr/bin/env python3
"""Build PaddleOCR FinTabNet.c splits from pre-cropped table images.

This is for Kaggle mirrors such as lewisdo/fintabnet, which provide
FinTabNet.c-Structure/images/*.jpg. It avoids downloading SEC PDFs and rendering
pages. Cell PDF boxes are mapped into cropped-image pixels by normalizing within
pdf_table_bbox.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import tarfile
from pathlib import Path

from PIL import Image

from build_fintabnet_c_paddleocr import (
    cell_text,
    clamp_box,
    company_from_table,
    iter_tables_from_tar,
    select_tables,
    structure_length,
    structure_tokens_and_cells,
    table_pdf_rel,
)


def image_stem(table: dict) -> str:
    rel = Path(table_pdf_rel(table))
    company = rel.parts[0] if len(rel.parts) >= 3 else company_from_table(table)
    year = rel.parts[1] if len(rel.parts) >= 3 else "unknown"
    page = rel.stem
    table_idx = table.get("document_table_index", 0)
    return f"{company}_{year}_{page}_table_{table_idx}"


def find_image(table: dict, image_root: Path, exts: list[str]) -> Path | None:
    stem = image_stem(table)
    for ext in exts:
        p = image_root / f"{stem}.{ext.lstrip('.')}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def norm_box(box):
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return left, top, right, bottom


def map_pdf_box_to_image(cell_box, table_box, width: int, height: int):
    tl, tt, tr, tb = norm_box(table_box)
    cl, ct, cr, cb = norm_box(cell_box)
    tw = max(tr - tl, 1e-6)
    th = max(tb - tt, 1e-6)
    mapped = [
        (cl - tl) / tw * width,
        (ct - tt) / th * height,
        (cr - tl) / tw * width,
        (cb - tt) / th * height,
    ]
    return clamp_box(mapped, width, height)


def convert_table(table: dict, src_image: Path, out_image_name: str, copy_images: bool, image_dir: Path):
    with Image.open(src_image) as im:
        width, height = im.size
    tokens, cells = structure_tokens_and_cells(table)
    table_box = table.get("pdf_table_bbox")
    out_cells = []
    for cell in cells:
        out_cell = dict(cell)
        pdf_bbox = out_cell.pop("pdf_bbox", None)
        if table_box and isinstance(pdf_bbox, list) and len(pdf_bbox) >= 4:
            out_cell["bbox"] = map_pdf_box_to_image(pdf_bbox, table_box, width, height)
        out_cells.append(out_cell)

    dst = image_dir / out_image_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        if copy_images:
            shutil.copy2(src_image, dst)
        else:
            try:
                dst.symlink_to(src_image.resolve())
            except FileExistsError:
                pass
            except OSError:
                shutil.copy2(src_image, dst)

    return {
        "filename": f"images/{out_image_name}",
        "html": {"structure": {"tokens": tokens}, "cells": out_cells},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-tar", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sampling", choices=["sequential", "round_robin_company"], default="round_robin_company")
    parser.add_argument("--max-structure-len", type=int, default=500)
    parser.add_argument("--extensions", default="jpg,jpeg,png")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    annotations_tar = Path(args.annotations_tar)
    image_root = Path(args.image_root)
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    exts = [x.strip().lstrip(".") for x in args.extensions.split(",") if x.strip()]

    source = []
    filtered = []
    for table in iter_tables_from_tar(annotations_tar, args.split_name):
        reason = None
        if args.max_structure_len and structure_length(table) > args.max_structure_len:
            reason = "structure_too_long"
        elif find_image(table, image_root, exts) is None:
            reason = "missing_image"
        if reason:
            filtered.append({"pdf": table_pdf_rel(table), "image_stem": image_stem(table), "reason": reason})
            continue
        source.append(table)

    selected = select_tables(source, args.limit, args.sampling)
    label_path = out_dir / f"fintabnet_c_{args.split_name}_{args.sampling}_{args.limit}_from_images.jsonl"
    manifest_path = out_dir / f"fintabnet_c_{args.split_name}_{args.sampling}_{args.limit}_from_images_manifest.json"

    converted = []
    skipped = []
    for index, table in enumerate(selected):
        src_img = find_image(table, image_root, exts)
        if src_img is None:
            skipped.append({"pdf": table_pdf_rel(table), "image_stem": image_stem(table), "error": "missing_image_after_select"})
            continue
        out_name = f"{image_stem(table)}{src_img.suffix.lower()}"
        try:
            converted.append(convert_table(table, src_img, out_name, args.copy_images, image_dir))
            print(f"[{len(converted):04d}] {src_img} -> images/{out_name}")
        except Exception as exc:
            skipped.append({"pdf": table_pdf_rel(table), "image_stem": image_stem(table), "error": repr(exc)})
            print(f"[skip] {table_pdf_rel(table)}: {exc}")

    with label_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "annotations_tar": str(annotations_tar),
        "image_root": str(image_root),
        "output_dir": str(out_dir),
        "label_file": str(label_path),
        "split": args.split_name,
        "sampling": args.sampling,
        "limit": args.limit,
        "max_structure_len": args.max_structure_len,
        "source_after_filter": len(source),
        "filtered_count": len(filtered),
        "selected": len(selected),
        "converted": len(converted),
        "skipped": skipped,
        "selected_company_counts": collections.Counter(company_from_table(t) for t in selected),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
