#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                record = json.loads(line)
                record["_line_no"] = line_no
                yield record


def score_record(record):
    tags = record.get("tags") or {}
    features = record.get("features") or {}
    cells = record.get("cells") or []
    text_boxes = record.get("text_boxes") or []

    score = 0
    if tags.get("has_merged_cells"):
        score += 100
    if tags.get("dense_table"):
        score += 80
    if tags.get("hard_case"):
        score += 60
    if tags.get("has_multiline_text"):
        score += 30
    score += min(len(cells), 250) / 10
    score += min(len(text_boxes), 200) / 20
    score += int(features.get("empty_cell_count", 0) or 0) / 5
    return score


def choose_pilot(records, limit):
    ranked = sorted(records, key=lambda r: (-score_record(r), r.get("filename", "")))
    selected = []
    seen = set()

    def add_matching(predicate):
        for record in ranked:
            name = record.get("filename")
            if name in seen or not predicate(record):
                continue
            selected.append(record)
            seen.add(name)
            if len(selected) >= limit:
                return

    add_matching(lambda r: (r.get("tags") or {}).get("has_merged_cells"))
    add_matching(lambda r: (r.get("tags") or {}).get("dense_table"))
    add_matching(lambda r: (r.get("tags") or {}).get("hard_case"))
    add_matching(lambda r: True)
    return selected[:limit]


def strip_internal(record):
    record = dict(record)
    record.pop("_line_no", None)
    return record


def summarize(records):
    tag_counts = Counter()
    total_cells = 0
    total_text_boxes = 0
    checked = 0
    review = 0
    for record in records:
        tags = record.get("tags") or {}
        for key, value in tags.items():
            if isinstance(value, bool) and value:
                tag_counts[key] += 1
            elif value not in (False, None, "", "unknown"):
                tag_counts[f"{key}:{value}"] += 1
        total_cells += len(record.get("cells") or [])
        total_text_boxes += len(record.get("text_boxes") or [])
        quality = record.get("quality") or {}
        checked += int(bool(quality.get("checked")))
        review += int(bool(quality.get("needs_human_review")))

    return {
        "samples": len(records),
        "cells": total_cells,
        "text_boxes": total_text_boxes,
        "checked_records": checked,
        "needs_human_review_records": review,
        "tag_counts": dict(tag_counts),
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(strip_internal(record), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    records = list(load_jsonl(args.input))
    selected = choose_pilot(records, args.limit)

    for record in selected:
        generation = record.setdefault("generation", {})
        generation["pilot_subset"] = "pilot_100"
        generation["pilot_selection_score"] = score_record(record)
        quality = record.setdefault("quality", {})
        quality.setdefault("checked", False)
        quality.setdefault("needs_human_review", True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, selected)

    with open(args.manifest, "w", encoding="utf-8") as f:
        for i, record in enumerate(selected):
            row = {
                "pilot_index": i,
                "source_line_no": record.get("_line_no"),
                "filename": record.get("filename"),
                "image_path": record.get("image_path"),
                "tags": record.get("tags"),
                "cells": len(record.get("cells") or []),
                "text_boxes": len(record.get("text_boxes") or []),
                "selection_score": score_record(record),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(selected)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(args.quality_report, "w", encoding="utf-8") as f:
        f.write("# PubTabNet-CTA Pilot 100 Quality Report\n\n")
        f.write("This pilot subset is selected for manual correction and metric validation.\n")
        f.write("The current records remain pseudo-labeled unless `quality.checked=true` after annotation.\n\n")
        for key, value in summary.items():
            f.write(f"- {key}: {value}\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
