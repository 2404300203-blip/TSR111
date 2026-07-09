#!/usr/bin/env python3
"""Prepare FinTabNet/FinTabNet.c splits for PaddleOCR table experiments.

The script is intentionally conservative: it does not assume one fixed
FinTabNet mirror layout. It accepts JSONL files in the common FinTabNet style
and writes PaddleOCR-compatible JSONL records plus company-disjoint optional
subsets when company metadata is available.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}") from exc


def get_nested(record, keys, default=None):
    cur = record
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def tokens_to_structure(record):
    html = record.get("html") or {}
    structure = html.get("structure") or {}
    tokens = structure.get("tokens")
    if isinstance(tokens, list):
        return tokens
    if isinstance(record.get("structure"), list):
        return record["structure"]
    if isinstance(record.get("tokens"), list):
        return record["tokens"]
    return None


def cell_tokens(record):
    html = record.get("html") or {}
    cells = html.get("cells")
    if isinstance(cells, list):
        return cells
    cells = record.get("cells")
    if isinstance(cells, list):
        return cells
    return []


def normalize_bbox(cell):
    bbox = (
        cell.get("bbox")
        or cell.get("box")
        or cell.get("rect")
        or cell.get("cell_box")
        or cell.get("pdf_bbox")
    )
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    vals = [float(v) for v in bbox[:4]]
    x1, y1, x2, y2 = vals
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])
    return [left, top, right, bottom]


def infer_filename(record):
    for key in ("filename", "image_path", "img_path", "file_name", "table_image"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    doc_id = record.get("doc_id") or record.get("filename_id") or record.get("imgid")
    table_id = record.get("table_id") or record.get("table_idx") or record.get("id")
    if doc_id is not None and table_id is not None:
        return f"{doc_id}_{table_id}.png"
    return None


def infer_company(record, filename):
    for key in ("company", "company_name", "ticker", "cik"):
        value = record.get(key)
        if value:
            return str(value)
    for path in (("metadata", "company"), ("metadata", "ticker"), ("metadata", "cik")):
        value = get_nested(record, path)
        if value:
            return str(value)
    # FinTabNet filenames often contain document identifiers; this fallback is
    # only for stable splitting when explicit company metadata is unavailable.
    if filename:
        stem = Path(filename).stem
        return re.split(r"[_\\-]", stem)[0]
    return "unknown"


def convert_record(record, split, image_prefix=None):
    structure = tokens_to_structure(record)
    if not structure:
        return None
    filename = infer_filename(record)
    if not filename:
        return None
    if image_prefix:
        filename = str(Path(image_prefix) / filename)
    cells = cell_tokens(record)
    bboxes = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        bbox = normalize_bbox(cell)
        if bbox is not None:
            bboxes.append(bbox)
    converted = {
        "filename": filename,
        "split": split,
        "html": {
            "structure": {"tokens": structure},
            "cells": cells,
        },
    }
    if bboxes:
        converted["html"]["cells"] = cells
    return converted


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} -> {path}")


def load_split(path: Path, split: str, image_prefix=None, limit=0):
    rows = []
    companies = []
    skipped = 0
    for _, record in read_jsonl(path):
        row = convert_record(record, split, image_prefix=image_prefix)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        companies.append(infer_company(record, row["filename"]))
        if limit and len(rows) >= limit:
            break
    return rows, companies, skipped


def company_disjoint_split(rows, companies, train_companies=50, test_companies=50, seed=42):
    rng = random.Random(seed)
    unique = sorted(set(companies))
    rng.shuffle(unique)
    test_set = set(unique[:test_companies])
    train_set = set(unique[test_companies : test_companies + train_companies])
    train_rows, test_rows = [], []
    for row, company in zip(rows, companies):
        if company in test_set:
            test_rows.append(row)
        elif company in train_set:
            train_rows.append(row)
    return train_rows, test_rows, sorted(train_set), sorted(test_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-name", required=True, help="fintabnet_original or fintabnet_c")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--test-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-prefix", default="")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--company-train-count", type=int, default=50)
    parser.add_argument("--company-test-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    train_rows, train_companies, train_skipped = load_split(
        Path(args.train_jsonl), "train", args.image_prefix, args.train_limit
    )
    val_rows, val_companies, val_skipped = load_split(
        Path(args.val_jsonl), "val", args.image_prefix, args.val_limit
    )
    if args.test_jsonl:
        test_rows, test_companies, test_skipped = load_split(
            Path(args.test_jsonl), "test", args.image_prefix, args.test_limit
        )
    else:
        test_rows, test_companies, test_skipped = [], [], 0

    write_jsonl(out / f"{args.version_name}_train_paddleocr.jsonl", train_rows)
    write_jsonl(out / f"{args.version_name}_val_paddleocr.jsonl", val_rows)
    if test_rows:
        write_jsonl(out / f"{args.version_name}_test_paddleocr.jsonl", test_rows)

    all_rows = train_rows + val_rows + test_rows
    all_companies = train_companies + val_companies + test_companies
    cd_train, cd_test, train_set, test_set = company_disjoint_split(
        all_rows,
        all_companies,
        train_companies=args.company_train_count,
        test_companies=args.company_test_count,
        seed=args.seed,
    )
    write_jsonl(out / f"{args.version_name}_company_disjoint_train.jsonl", cd_train)
    write_jsonl(out / f"{args.version_name}_company_disjoint_test.jsonl", cd_test)
    manifest = {
        "version_name": args.version_name,
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "company_disjoint_train": len(cd_train),
            "company_disjoint_test": len(cd_test),
        },
        "skipped": {
            "train": train_skipped,
            "val": val_skipped,
            "test": test_skipped,
        },
        "company_split": {
            "train_companies": train_set,
            "test_companies": test_set,
        },
    }
    (out / f"{args.version_name}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
