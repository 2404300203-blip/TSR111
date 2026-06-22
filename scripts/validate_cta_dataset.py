#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def is_bbox(value):
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0


def is_zero_bbox(value):
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return False
    return x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0


def is_empty_cell(cell):
    text_empty = not (cell.get("text") or "").strip()
    tokens = cell.get("tokens") or []
    token_text = "".join(str(token) for token in tokens)
    tokens_empty = not token_text.replace("<b>", "").replace("</b>", "").strip()
    return text_empty and tokens_empty


def validate_record(record, line_no):
    errors = []
    warnings = []
    for key in ("filename", "image_path", "image_size", "cells", "text_boxes", "quality"):
        if key not in record:
            errors.append(f"missing {key}")

    image_size = record.get("image_size") or {}
    width = image_size.get("width")
    height = image_size.get("height")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        errors.append("invalid image_size")

    cell_ids = set()
    for idx, cell in enumerate(record.get("cells") or []):
        cell_id = cell.get("cell_id")
        if not cell_id:
            errors.append(f"cell[{idx}] missing cell_id")
        else:
            cell_ids.add(cell_id)
        bbox = cell.get("bbox")
        if not is_bbox(bbox):
            if is_empty_cell(cell) and (bbox is None or is_zero_bbox(bbox)):
                warnings.append(f"cell[{idx}] empty cell has missing/zero bbox")
            else:
                errors.append(f"cell[{idx}] invalid bbox")

    for idx, text_box in enumerate(record.get("text_boxes") or []):
        if not text_box.get("text_id"):
            errors.append(f"text_boxes[{idx}] missing text_id")
        if not is_bbox(text_box.get("bbox")):
            errors.append(f"text_boxes[{idx}] invalid bbox")
        cell_id = text_box.get("cell_id")
        if cell_id and cell_id not in cell_ids:
            errors.append(f"text_boxes[{idx}] references unknown cell_id {cell_id}")

    quality = record.get("quality") or {}
    if "checked" not in quality:
        errors.append("quality.checked missing")
    if "needs_human_review" not in quality:
        errors.append("quality.needs_human_review missing")

    image_path = record.get("image_path")
    if image_path and not Path(image_path).exists():
        errors.append(f"image_path does not exist: {image_path}")

    return [f"line {line_no}: {e}" for e in errors], [
        f"line {line_no}: {w}" for w in warnings
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--max-errors", type=int, default=50)
    args = parser.parse_args()

    errors = []
    warnings = []
    count = 0
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: json decode error {exc}")
                continue
            record_errors, record_warnings = validate_record(record, line_no)
            errors.extend(record_errors)
            warnings.extend(record_warnings)
            if len(errors) >= args.max_errors:
                break

    print(
        json.dumps(
            {"records": count, "errors": len(errors), "warnings": len(warnings)},
            indent=2,
        )
    )
    for error in errors[: args.max_errors]:
        print(error)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
