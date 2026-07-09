#!/usr/bin/env python3
"""Build a PaddleOCR-ready split from FinTabNet.c annotations.

FinTabNet.c stores table structure as per-page JSON annotations with PDF
coordinates measured from the top-left page corner. This script downloads the
corresponding FinTabNet PDF pages from Kaggle, renders each page, crops the
table region, reconstructs simple PubTabNet-style HTML structure tokens, and
writes a PaddleOCR/PubTabNet-compatible JSONL label file.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from PIL import Image


DEFAULT_PDFTOPPM = (
    "/Users/table/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pdftoppm"
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def company_from_table(table: dict) -> str:
    folder = str(table.get("pdf_folder") or "")
    return folder.strip("/").split("/", 1)[0] if folder else "unknown"


def table_pdf_rel(table: dict) -> str:
    folder = str(table.get("pdf_folder") or "").strip("/")
    name = str(table.get("pdf_file_name") or "")
    return f"{folder}/{name}" if folder else name


def cell_text(cell: dict) -> str:
    return str(
        cell.get("pdf_text_content")
        or cell.get("json_text_content")
        or ""
    )


def td_open_tokens(rowspan: int, colspan: int) -> list[str]:
    attrs = []
    if rowspan > 1:
        attrs.append(f' rowspan="{rowspan}"')
    if colspan > 1:
        attrs.append(f' colspan="{colspan}"')
    if not attrs:
        return ["<td>"]
    return ["<td", *attrs, ">"]


def structure_tokens_and_cells(table: dict) -> tuple[list[str], list[dict]]:
    raw_cells = []
    for cell in table.get("cells", []):
        rows = sorted(int(x) for x in cell.get("row_nums", []))
        cols = sorted(int(x) for x in cell.get("column_nums", []))
        if not rows or not cols:
            continue
        raw_cells.append(
            {
                "row0": rows[0],
                "col0": cols[0],
                "rowspan": rows[-1] - rows[0] + 1,
                "colspan": cols[-1] - cols[0] + 1,
                "rows": rows,
                "cols": cols,
                "cell": cell,
            }
        )

    if not raw_cells:
        return ["<table>", "</table>"], []

    max_row = max(max(item["rows"]) for item in raw_cells)
    max_col = max(max(item["cols"]) for item in raw_cells)
    starts = {(item["row0"], item["col0"]): item for item in raw_cells}
    covered = set()
    for item in raw_cells:
        for r in item["rows"]:
            for c in item["cols"]:
                if (r, c) != (item["row0"], item["col0"]):
                    covered.add((r, c))

    tokens = ["<table>"]
    out_cells = []
    for row in range(max_row + 1):
        tokens.append("<tr>")
        for col in range(max_col + 1):
            if (row, col) in covered:
                continue
            item = starts.get((row, col))
            if item is None:
                tokens.extend(["<td>", "</td>"])
                out_cells.append({"tokens": []})
                continue
            tokens.extend(td_open_tokens(item["rowspan"], item["colspan"]))
            tokens.append("</td>")
            source_cell = item["cell"]
            out_cell = {
                "tokens": list(cell_text(source_cell)),
            }
            if "pdf_bbox" in source_cell:
                out_cell["pdf_bbox"] = source_cell["pdf_bbox"]
            out_cells.append(out_cell)
        tokens.append("</tr>")
    tokens.append("</table>")
    return tokens, out_cells


def structure_length(table: dict) -> int:
    tokens, _ = structure_tokens_and_cells(table)
    return len(tokens)


def select_tables(tables: list[dict], limit: int, sampling: str) -> list[dict]:
    if limit <= 0 or len(tables) <= limit:
        return tables
    if sampling == "sequential":
        return tables[:limit]
    if sampling != "round_robin_company":
        raise ValueError(f"unsupported sampling mode: {sampling}")

    grouped = collections.defaultdict(list)
    for table in tables:
        grouped[company_from_table(table)].append(table)
    selected = []
    companies = sorted(grouped)
    cursor = 0
    while len(selected) < limit and companies:
        company = companies[cursor % len(companies)]
        bucket = grouped[company]
        if bucket:
            selected.append(bucket.pop(0))
        if not bucket:
            companies.remove(company)
            if not companies:
                break
            cursor %= len(companies)
        else:
            cursor += 1
    return selected


def iter_tables_from_tar(annotations_tar: Path, split: str):
    with tarfile.open(annotations_tar, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                page_tables = json.loads(f.read().decode("utf-8"))
            except Exception as exc:
                print(f"[skip-json] {member.name}: {exc}")
                continue
            for table in page_tables:
                if table.get("split") == split:
                    yield table


def download_pdf(
    record_filename: str,
    pdf_root: Path,
    kaggle_bin: str,
    retries: int,
    retry_sleep: float,
) -> Path:
    rel = Path(record_filename)
    pdf_path = pdf_root / rel
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
    for attempt in range(1, retries + 1):
        try:
            run(cmd)
            break
        except subprocess.CalledProcessError:
            if attempt >= retries:
                raise
            wait = retry_sleep * attempt
            print(f"[retry {attempt}/{retries}] {record_filename}; sleep {wait:.1f}s")
            time.sleep(wait)
    downloaded = pdf_path.parent / rel.name
    if not downloaded.exists():
        raise FileNotFoundError(f"Kaggle did not create expected PDF: {downloaded}")
    return downloaded


def cached_pdf_path(record_filename: str, pdf_root: Path) -> Path:
    return pdf_root / Path(record_filename)


def render_pdf(pdf_path: Path, pdftoppm_bin: str, dpi: int) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "page"
        run([pdftoppm_bin, "-r", str(dpi), "-png", "-singlefile", str(pdf_path), str(out_prefix)])
        image_path = out_prefix.with_suffix(".png")
        with Image.open(image_path) as im:
            return im.convert("RGB")


def top_origin_pdf_box_to_pixels(box, scale: float):
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return [left * scale, top * scale, right * scale, bottom * scale]


def clamp_box(box, width: int, height: int):
    left, top, right, bottom = box
    left = max(0, min(width - 1, int(round(left))))
    top = max(0, min(height - 1, int(round(top))))
    right = max(left + 1, min(width, int(round(right))))
    bottom = max(top + 1, min(height, int(round(bottom))))
    return [left, top, right, bottom]


def convert_table(table: dict, image: Image.Image, image_name: str, dpi: int):
    scale = dpi / 72.0
    table_box_px = clamp_box(
        top_origin_pdf_box_to_pixels(table["pdf_table_bbox"], scale),
        image.width,
        image.height,
    )
    crop = image.crop(tuple(table_box_px))
    crop_w, crop_h = crop.size
    table_left, table_top, _, _ = table_box_px

    tokens, cells = structure_tokens_and_cells(table)
    out_cells = []
    for cell in cells:
        out_cell = dict(cell)
        pdf_bbox = out_cell.pop("pdf_bbox", None)
        if isinstance(pdf_bbox, list) and len(pdf_bbox) >= 4:
            cell_px = top_origin_pdf_box_to_pixels(pdf_bbox, scale)
            rel = [
                cell_px[0] - table_left,
                cell_px[1] - table_top,
                cell_px[2] - table_left,
                cell_px[3] - table_top,
            ]
            out_cell["bbox"] = clamp_box(rel, crop_w, crop_h)
        out_cells.append(out_cell)

    out_record = {
        "filename": f"images/{image_name}",
        "html": {
            "structure": {"tokens": tokens},
            "cells": out_cells,
        },
    }
    return crop, out_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-tar", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sampling", choices=["sequential", "round_robin_company"], default="sequential")
    parser.add_argument("--max-structure-len", type=int, default=500)
    parser.add_argument("--exclude-detection", action="store_true")
    parser.add_argument("--exclude-structure", action="store_true")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--pdf-root", default="")
    parser.add_argument("--kaggle-bin", default=str(Path.home() / ".local/bin/kaggle"))
    parser.add_argument("--pdftoppm-bin", default=DEFAULT_PDFTOPPM)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--require-existing-pdf",
        action="store_true",
        help="Only select tables whose PDF already exists under --pdf-root.",
    )
    args = parser.parse_args()

    annotations_tar = Path(args.annotations_tar)
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images"
    pdf_root = Path(args.pdf_root) if args.pdf_root else out_dir / "pdf"
    tag = f"{args.split_name}_{args.sampling}_{args.limit}"
    label_path = out_dir / f"fintabnet_c_{tag}.jsonl"
    manifest_path = out_dir / f"fintabnet_c_{tag}_manifest.json"
    image_dir.mkdir(parents=True, exist_ok=True)
    pdf_root.mkdir(parents=True, exist_ok=True)

    source_tables = []
    filtered_out = []
    for table in iter_tables_from_tar(annotations_tar, args.split_name):
        reason = None
        if args.exclude_detection and table.get("exclude_for_detection"):
            reason = "exclude_for_detection"
        elif args.exclude_structure and table.get("exclude_for_structure"):
            reason = "exclude_for_structure"
        elif args.max_structure_len and structure_length(table) > args.max_structure_len:
            reason = "structure_too_long"
        elif args.require_existing_pdf and not cached_pdf_path(table_pdf_rel(table), pdf_root).exists():
            reason = "missing_cached_pdf"

        if reason:
            filtered_out.append(
                {
                    "structure_id": table.get("structure_id"),
                    "pdf": table_pdf_rel(table),
                    "reason": reason,
                }
            )
            continue
        source_tables.append(table)

    selected_tables = select_tables(source_tables, args.limit, args.sampling)

    converted = []
    skipped = []
    for index, table in enumerate(selected_tables):
        pdf_rel = table_pdf_rel(table)
        try:
            pdf_path = download_pdf(
                pdf_rel,
                pdf_root,
                args.kaggle_bin,
                args.download_retries,
                args.retry_sleep,
            )
            page = render_pdf(pdf_path, args.pdftoppm_bin, args.dpi)
            structure_id = table.get("structure_id") or Path(pdf_rel).with_suffix("").as_posix().replace("/", "_")
            image_name = f"{structure_id}_table_{table.get('document_table_index', index)}.png"
            crop, out_record = convert_table(table, page, image_name, args.dpi)
            crop.save(image_dir / image_name)
            converted.append(out_record)
            print(f"[{len(converted):04d}] {pdf_rel} -> images/{image_name}")
        except Exception as exc:
            skipped.append({"pdf": pdf_rel, "structure_id": table.get("structure_id"), "error": repr(exc)})
            print(f"[skip] {pdf_rel}: {exc}")

    with label_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "annotations_tar": str(annotations_tar),
        "output_dir": str(out_dir),
        "label_file": str(label_path),
        "image_dir": str(image_dir),
        "pdf_root": str(pdf_root),
        "dpi": args.dpi,
        "split": args.split_name,
        "sampling": args.sampling,
        "max_structure_len": args.max_structure_len,
        "exclude_detection": args.exclude_detection,
        "exclude_structure": args.exclude_structure,
        "download_retries": args.download_retries,
        "retry_sleep": args.retry_sleep,
        "require_existing_pdf": args.require_existing_pdf,
        "source_tables_after_filter": len(source_tables),
        "selected_tables": len(selected_tables),
        "converted": len(converted),
        "filtered_out_count": len(filtered_out),
        "selected_company_counts": collections.Counter(
            company_from_table(table) for table in selected_tables
        ),
        "skipped": skipped,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
