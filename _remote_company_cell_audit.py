import json
import re
import sys
from pathlib import Path


SKIP = {"<html>", "</html>", "<body>", "</body>", "<thead>", "</thead>", "<tbody>", "</tbody>"}


def merge_tag_tokens(tokens):
    merged = []
    i = 0
    while i < len(tokens):
        token = str(tokens[i])
        if token.startswith("<td") or token.startswith("<th"):
            value = token
            i += 1
            while i < len(tokens) and ">" not in value:
                value += str(tokens[i])
                i += 1
            merged.append(value)
        else:
            merged.append(token)
            i += 1
    return merged


def parse_cells(tokens):
    cells = set()
    occupied = set()
    row = -1
    col = 0
    in_row = False
    cleaned = [str(x) for x in tokens if str(x) not in SKIP]
    for token in merge_tag_tokens(cleaned):
        if token.startswith("<tr"):
            row += 1
            col = 0
            in_row = True
            continue
        if token.startswith("</tr"):
            in_row = False
            continue
        if not in_row or not (token.startswith("<td") or token.startswith("<th")):
            continue
        while (row, col) in occupied:
            col += 1
        row_match = re.search(r'rowspan=["\']([0-9]+)["\']', token)
        col_match = re.search(r'colspan=["\']([0-9]+)["\']', token)
        rowspan = int(row_match.group(1)) if row_match else 1
        colspan = int(col_match.group(1)) if col_match else 1
        cells.add((row, col, rowspan, colspan))
        for rr in range(row, row + rowspan):
            for cc in range(col, col + colspan):
                occupied.add((rr, cc))
        col += colspan
    return cells


def audit(path):
    tp = fp = fn = 0
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        pred = parse_cells(item.get("pred_tokens", []))
        gt = parse_cells(item.get("gt_tokens", []))
        row_tp = len(pred & gt)
        row_fp = len(pred - gt)
        row_fn = len(gt - pred)
        tp += row_tp
        fp += row_fp
        fn += row_fn
        denom = 2 * row_tp + row_fp + row_fn
        rows.append({
            "sample_index": item.get("sample_index"),
            "filename": item.get("filename", ""),
            "tp": row_tp,
            "fp": row_fp,
            "fn": row_fn,
            "cell_f1": 2 * row_tp / denom if denom else 1.0,
            "pred_cells": len(pred),
            "gt_cells": len(gt),
        })
    denom = 2 * tp + fp + fn
    return {
        "source": str(path),
        "samples": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "cell_f1_micro": 2 * tp / denom if denom else 1.0,
        "cell_f1_macro": sum(row["cell_f1"] for row in rows) / max(len(rows), 1),
        "rows": rows,
    }


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        result = audit(filename)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
