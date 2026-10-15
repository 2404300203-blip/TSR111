#!/usr/bin/env python3
"""Post-hoc CTA-500 header-focused structural evaluation.

The evaluator consumes frozen sample-level predictions.  Header depth is read
only from the GT ``<thead>`` region and is never exposed to the model.  Both
prediction and GT are evaluated over their first D_H logical rows; cell spans
are retained unchanged so a predicted span leaking from header into body is
still penalized.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "remote_results" / "main_tables_rebuild_20260811" / "sample_jsonl"
CTA500_MANIFEST = ROOT / "remote_results" / "main_tables_rebuild_20260811" / "complex_structure_cta500_20260625.jsonl"
CTA500_TAGS = ROOT / "论文材料" / "kbs_appendix_experiments_20260803" / "tables" / "A00_cta500_merged_sample_records.csv"
OUTPUT = ROOT / "remote_results" / "header_metrics_20260814"

MODELS = {
    "slanet": "SLANet",
    "cell_token": "Cell Token only",
    "true_horizontal": "True Horizontal-only",
    "true_vertical": "True Vertical-only",
    "dbm_slanet": "DBM-SLANet",
}


@dataclass
class Node:
    label: str
    children: list["Node"] = field(default_factory=list)


def reconstructed_tokens(tokens: Iterable[str]) -> list[str]:
    """Join split opening tags emitted by PaddleOCR's structure decoder."""
    values = [str(token) for token in tokens]
    output: list[str] = []
    index = 0
    while index < len(values):
        token = values[index]
        if token.startswith("<td") or token.startswith("<th"):
            pieces = [token]
            while ">" not in "".join(pieces) and index + 1 < len(values):
                index += 1
                pieces.append(values[index])
            output.append("".join(pieces))
        else:
            output.append(token)
        index += 1
    return output


def is_open(token: str, tag: str) -> bool:
    value = token.strip().lower()
    return value.startswith(f"<{tag}") and not value.startswith(f"</{tag}")


def is_close(token: str, tag: str) -> bool:
    return token.strip().lower().startswith(f"</{tag}")


def header_depth(gt_tokens: Iterable[str]) -> int | None:
    tokens = reconstructed_tokens(gt_tokens)
    start = next((i for i, token in enumerate(tokens) if is_open(token, "thead")), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(tokens)) if is_close(tokens[i], "thead")), None)
    if end is None:
        return None
    return sum(is_open(token, "tr") for token in tokens[start + 1 : end])


def thead_tokens(tokens: Iterable[str]) -> list[str] | None:
    """Return a normalized complete thead region, if one is present."""
    values = reconstructed_tokens(tokens)
    start = next((i for i, token in enumerate(values) if is_open(token, "thead")), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(values)) if is_close(values[i], "thead")), None)
    return values[start : end + 1] if end is not None else None


def first_rows(tokens: Iterable[str], depth: int) -> list[str]:
    """Create a synthetic thead containing the first logical table rows."""
    rows: list[list[str]] = []
    active: list[str] | None = None
    for token in reconstructed_tokens(tokens):
        if is_open(token, "tr"):
            if active is None:
                active = [token]
            continue
        if active is None:
            continue
        active.append(token)
        if is_close(token, "tr"):
            rows.append(active)
            active = None
            if len(rows) >= depth:
                break
    flattened = ["<thead>"]
    for row in rows:
        flattened.extend(row)
    flattened.append("</thead>")
    return flattened


def normalize_token(token: str) -> str:
    token = str(token).strip()
    if token.startswith("</") or (token.startswith("<") and token.endswith(">")):
        return token
    return "#text"


def build_tree(tokens: Iterable[str]) -> Node:
    root = Node("root")
    stack = [root]
    for raw in tokens:
        token = normalize_token(raw)
        if not token:
            continue
        if token.startswith("</"):
            if len(stack) > 1:
                stack.pop()
            continue
        node = Node(token)
        stack[-1].children.append(node)
        if token.startswith("<") and token not in {"<br>", "<hr>", "<img>"}:
            stack.append(node)
    return root


def tree_size(node: Node) -> int:
    return 1 + sum(tree_size(child) for child in node.children)


def tree_distance(a: Node, b: Node, memo: dict[tuple[int, int], int]) -> int:
    key = (id(a), id(b))
    if key in memo:
        return memo[key]
    rename = int(a.label != b.label)
    dp = [[0] * (len(b.children) + 1) for _ in range(len(a.children) + 1)]
    for i in range(1, len(a.children) + 1):
        dp[i][0] = dp[i - 1][0] + tree_size(a.children[i - 1])
    for j in range(1, len(b.children) + 1):
        dp[0][j] = dp[0][j - 1] + tree_size(b.children[j - 1])
    for i, child_a in enumerate(a.children, 1):
        for j, child_b in enumerate(b.children, 1):
            dp[i][j] = min(
                dp[i - 1][j] + tree_size(child_a),
                dp[i][j - 1] + tree_size(child_b),
                dp[i - 1][j - 1] + tree_distance(child_a, child_b, memo),
            )
    memo[key] = rename + dp[-1][-1]
    return memo[key]


def teds(tokens_pred: Iterable[str], tokens_gt: Iterable[str]) -> float:
    pred, gt = build_tree(tokens_pred), build_tree(tokens_gt)
    denominator = max(tree_size(pred), tree_size(gt), 1)
    return max(0.0, 1.0 - tree_distance(pred, gt, {}) / denominator)


def span_value(tag: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*=\s*['\"]?(\d+)", tag, flags=re.I)
    return max(1, int(match.group(1))) if match else 1


def parse_cells(tokens: Iterable[str]) -> set[tuple[int, int, int, int]]:
    cells: set[tuple[int, int, int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    row, column, in_row = -1, 0, False
    for token in reconstructed_tokens(tokens):
        if is_open(token, "tr"):
            row, column, in_row = row + 1, 0, True
            continue
        if is_close(token, "tr"):
            in_row = False
            continue
        value = token.strip().lower()
        if not in_row or not (value.startswith("<td") or value.startswith("<th")):
            continue
        while (row, column) in occupied:
            column += 1
        rowspan, colspan = span_value(token, "rowspan"), span_value(token, "colspan")
        cells.add((row, column, rowspan, colspan))
        for rr in range(row, row + rowspan):
            for cc in range(column, column + colspan):
                occupied.add((rr, cc))
        column += colspan
    return cells


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cta_tag_groups() -> dict[str, set[str]]:
    groups = {"CTA-500 overall": set()}
    with CTA500_TAGS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filename = row["filename"]
            groups["CTA-500 overall"].add(filename)
            for tag in ast.literal_eval(row["tags"]):
                if tag in {"multi_level_header", "hierarchical_span_header"}:
                    groups.setdefault(tag, set()).add(filename)
    expected = {"CTA-500 overall": 500, "multi_level_header": 325, "hierarchical_span_header": 274}
    for group, count in expected.items():
        if len(groups.get(group, set())) != count:
            raise RuntimeError(f"{group}: expected {count} tagged samples, found {len(groups.get(group, set()))}")
    return groups


def score_model(model_id: str, label: str, group: str, allowed: set[str]) -> tuple[dict, list[dict]]:
    path = SOURCE / model_id / "sample_teds.jsonl"
    manifest = {
        json.loads(line)["filename"]
        for line in CTA500_MANIFEST.open(encoding="utf-8")
        if line.strip()
    }
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record["filename"] in manifest and record["filename"] in allowed:
            rows.append(record)
    selected = [row["filename"] for row in rows]
    if (
        len(manifest) != 500
        or len(rows) != len(allowed)
        or len(selected) != len(set(selected))
        or set(selected) != allowed
    ):
        raise RuntimeError(
            f"{model_id}: expected {len(allowed)} rows, found {len(rows)} "
            f"({len(set(selected))} unique) against a {len(manifest)}-file manifest"
        )
    header_scores: list[float] = []
    tp = fp = fn = 0
    audited: list[dict] = []
    missing_gt_header = missing_pred_header = 0

    for record in rows:
        depth = header_depth(record["gt_tokens"])
        if not depth:
            missing_gt_header += 1
            continue
        pred_has_header = header_depth(record["pred_tokens"])
        missing_pred_header += int(not pred_has_header)
        hh_teds = teds(first_rows(record["pred_tokens"], depth), first_rows(record["gt_tokens"], depth))
        header_scores.append(hh_teds)

        pred_spans = {
            cell for cell in parse_cells(record["pred_tokens"])
            if cell[0] < depth and (cell[2] > 1 or cell[3] > 1)
        }
        gt_spans = {
            cell for cell in parse_cells(record["gt_tokens"])
            if cell[0] < depth and (cell[2] > 1 or cell[3] > 1)
        }
        tp += len(pred_spans & gt_spans)
        fp += len(pred_spans - gt_spans)
        fn += len(gt_spans - pred_spans)
        audited.append({
            "model_id": model_id,
            "group": group,
            "sample_index": record.get("sample_index"),
            "filename": record.get("filename"),
            "header_depth": depth,
            "hh_teds": f"{hh_teds:.12f}",
            "pred_header_present": int(bool(pred_has_header)),
            "pred_header_spans": len(pred_spans),
            "gt_header_spans": len(gt_spans),
            "span_tp": len(pred_spans & gt_spans),
            "span_fp": len(pred_spans - gt_spans),
            "span_fn": len(gt_spans - pred_spans),
        })

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    hspan_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    summary = {
        "model": label,
        "id": model_id,
        "group": group,
        "n_total": len(rows),
        "n_header_evaluable": len(header_scores),
        "gt_missing_thead": missing_gt_header,
        "pred_missing_thead": missing_pred_header,
        "hh_teds": f"{sum(header_scores) / len(header_scores):.12f}",
        "hspan_precision": f"{precision:.12f}",
        "hspan_recall": f"{recall:.12f}",
        "hspan_f1": f"{hspan_f1:.12f}",
        "header_span_tp": tp,
        "header_span_fp": fp,
        "header_span_fn": fn,
    }
    return summary, audited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summaries, per_sample = [], []
    for group, filenames in cta_tag_groups().items():
        for model_id, label in MODELS.items():
            summary, audit = score_model(model_id, label, group, filenames)
            summaries.append(summary)
            per_sample.extend(audit)
    write_csv(args.output / "cta500_header_metrics.csv", summaries, list(summaries[0]))
    write_csv(args.output / "cta500_header_metrics_per_sample.csv", per_sample, list(per_sample[0]))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
