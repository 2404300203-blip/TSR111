#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


MODEL_IDS = [
    "slanet",
    "cell_token",
    "lite_mamba",
    "full_mamba_no_presence",
    "full_mamba_presence",
    "full_mamba_one_way",
    "dbm_d64",
    "dbm_ds8",
    "dbm_d192",
    "dbm_e1",
    "zero_geometry",
    "true_horizontal",
    "true_vertical",
    "dbm_slanet",
]


@dataclass
class Node:
    label: str
    children: list["Node"] = field(default_factory=list)


def normalize_token(token: str) -> str:
    token = str(token).strip()
    if token.startswith("</"):
        return token
    if token.startswith("<") and token.endswith(">"):
        return token
    return "#text"


def build_tree(tokens: list[str]) -> Node:
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
    ac, bc = a.children, b.children
    dp = [[0] * (len(bc) + 1) for _ in range(len(ac) + 1)]
    for i in range(1, len(ac) + 1):
        dp[i][0] = dp[i - 1][0] + tree_size(ac[i - 1])
    for j in range(1, len(bc) + 1):
        dp[0][j] = dp[0][j - 1] + tree_size(bc[j - 1])
    for i in range(1, len(ac) + 1):
        for j in range(1, len(bc) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + tree_size(ac[i - 1]),
                dp[i][j - 1] + tree_size(bc[j - 1]),
                dp[i - 1][j - 1] + tree_distance(ac[i - 1], bc[j - 1], memo),
            )
    value = (0 if a.label == b.label else 1) + dp[-1][-1]
    memo[key] = value
    return value


def teds_score(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    pred, gt = build_tree(pred_tokens), build_tree(gt_tokens)
    distance = tree_distance(pred, gt, {})
    return max(0.0, 1.0 - distance / max(tree_size(pred), tree_size(gt), 1))


def parse_span(token: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*=\s*['\"]?([0-9]+)", token, re.I)
    return max(1, int(match.group(1))) if match else 1


def header_depth(tokens: list[str]) -> int:
    in_header = False
    depth = 0
    for raw in tokens:
        token = str(raw).strip().lower()
        if token.startswith("<thead"):
            in_header = True
        elif token.startswith("</thead"):
            in_header = False
        elif in_header and token.startswith("<tr") and not token.startswith("</tr"):
            depth += 1
    return depth


def coalesce_tags(tokens: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(tokens):
        token = str(tokens[index]).strip()
        if (token.startswith("<td") or token.startswith("<th")) and ">" not in token:
            parts = [token]
            index += 1
            while index < len(tokens):
                part = str(tokens[index]).strip()
                parts.append(part)
                index += 1
                if ">" in part:
                    break
            result.append(" ".join(parts))
            continue
        result.append(token)
        index += 1
    return result


def parse_cells(tokens: list[str]) -> tuple[list[tuple[int, int, int, int]], int]:
    cells: list[tuple[int, int, int, int]] = []
    occupied: set[tuple[int, int]] = set()
    row = -1
    col = 0
    in_row = False
    row_count = 0
    for raw in coalesce_tags(tokens):
        token = str(raw).strip().lower()
        if token.startswith("<tr") and not token.startswith("</tr"):
            row += 1
            row_count += 1
            col = 0
            in_row = True
            continue
        if token.startswith("</tr"):
            in_row = False
            continue
        if in_row and (token.startswith("<td") or token.startswith("<th")):
            while (row, col) in occupied:
                col += 1
            rowspan = parse_span(token, "rowspan")
            colspan = parse_span(token, "colspan")
            cells.append((row, col, rowspan, colspan))
            for rr in range(row, row + rowspan):
                for cc in range(col, col + colspan):
                    occupied.add((rr, cc))
            col += colspan
    return cells, row_count


def header_tokens(cells: list[tuple[int, int, int, int]], depth: int) -> list[str]:
    result = ["<thead>"]
    by_row: dict[int, list[tuple[int, int, int, int]]] = {}
    for cell in cells:
        if cell[0] < depth:
            by_row.setdefault(cell[0], []).append(cell)
    for row in range(depth):
        result.append("<tr>")
        for _, _, rowspan, colspan in sorted(by_row.get(row, []), key=lambda x: x[1]):
            attrs = ""
            if rowspan > 1:
                attrs += f' rowspan="{rowspan}"'
            if colspan > 1:
                attrs += f' colspan="{colspan}"'
            result.append(f"<td{attrs}></td>")
        result.append("</tr>")
    result.append("</thead>")
    return result


def score_row(model: str, row: dict) -> dict:
    gt_depth = header_depth(row["gt_tokens"])
    pred_cells, pred_rows = parse_cells(row["pred_tokens"])
    gt_cells, gt_rows = parse_cells(row["gt_tokens"])
    status = "ok"
    if gt_depth <= 0:
        status = "missing_gt_thead"
    elif gt_rows < gt_depth:
        status = "malformed_gt_rows"
    elif pred_rows == 0:
        status = "malformed_pred_rows"
    pred_spans = {c for c in pred_cells if c[0] < gt_depth and (c[2] > 1 or c[3] > 1)}
    gt_spans = {c for c in gt_cells if c[0] < gt_depth and (c[2] > 1 or c[3] > 1)}
    tp = len(pred_spans & gt_spans)
    fp = len(pred_spans - gt_spans)
    fn = len(gt_spans - pred_spans)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gt_spans else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not pred_spans else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    pred_hash = hashlib.sha256(
        json.dumps(row["pred_tokens"], ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    gt_all_spans = {c for c in gt_cells if c[2] > 1 or c[3] > 1}
    return {
        "model": model,
        "sample_id": row["filename"],
        "prediction_sha256": pred_hash,
        "gt_header_depth": gt_depth,
        "parse_status": status,
        "hh_teds": teds_score(header_tokens(pred_cells, gt_depth), header_tokens(gt_cells, gt_depth)) if status == "ok" else 0.0,
        "header_span_tp": tp,
        "header_span_fp": fp,
        "header_span_fn": fn,
        "hspan_precision": precision,
        "hspan_recall": recall,
        "hspan_f1": f1,
        "gt_rowspan_count": sum(c[2] > 1 for c in gt_spans),
        "gt_colspan_count": sum(c[3] > 1 for c in gt_spans),
        "gt_has_any_span": int(bool(gt_all_spans)),
        "gt_has_header_span": int(bool(gt_spans)),
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def micro_f1(rows: list[dict]) -> tuple[float, float, float]:
    tp = sum(int(r["header_span_tp"]) for r in rows)
    fp = sum(int(r["header_span_fp"]) for r in rows)
    fn = sum(int(r["header_span_fn"]) for r in rows)
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return p, r, f


def summarize(model: str, rows: list[dict], file_path: Path) -> dict:
    p, r, f = micro_f1(rows)
    return {
        "model": model,
        "samples": len(rows),
        "parse_success": sum(x["parse_status"] == "ok" for x in rows),
        "hh_teds_mean": np.mean([x["hh_teds"] for x in rows]),
        "hspan_precision_micro": p,
        "hspan_recall_micro": r,
        "hspan_f1_micro": f,
        "hspan_f1_sample_mean": np.mean([x["hspan_f1"] for x in rows]),
        "gt_any_span_samples": sum(x["gt_has_any_span"] for x in rows),
        "gt_header_span_samples": sum(x["gt_has_header_span"] for x in rows),
        "prediction_file": str(file_path),
        "prediction_file_sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
    }


def bootstrap(rows_by_model: dict[str, list[dict]], reps: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(rows_by_model["slanet"])
    indices = rng.integers(0, n, size=(reps, n), endpoint=False)
    arrays = {}
    for model, rows in rows_by_model.items():
        arrays[model] = {
            "hh": np.asarray([r["hh_teds"] for r in rows], dtype=float),
            "tp": np.asarray([r["header_span_tp"] for r in rows], dtype=float),
            "fp": np.asarray([r["header_span_fp"] for r in rows], dtype=float),
            "fn": np.asarray([r["header_span_fn"] for r in rows], dtype=float),
        }
    result = []
    comparisons = [(m, "slanet") for m in MODEL_IDS if m != "slanet"]
    comparisons += [("dbm_slanet", m) for m in MODEL_IDS if m != "dbm_slanet"]
    for model, reference in comparisons:
        a, b = arrays[model], arrays[reference]
        hh_delta = (a["hh"][indices] - b["hh"][indices]).mean(axis=1)
        f1_delta = []
        for idx in indices:
            vals = []
            for x in (a, b):
                tp, fp, fn = x["tp"][idx].sum(), x["fp"][idx].sum(), x["fn"][idx].sum()
                vals.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0)
            f1_delta.append(vals[0] - vals[1])
        for metric, values in (("hh_teds", hh_delta), ("hspan_f1_micro", np.asarray(f1_delta))):
            result.append({
                "model": model,
                "reference": reference,
                "metric": metric,
                "bootstrap_replicates": reps,
                "observed_delta": float(values.mean()),
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "probability_delta_gt_zero": float(np.mean(values > 0)),
                "seed": seed,
            })
    return result


def run_unit_tests() -> list[dict]:
    tests = []
    def check(name: str, condition: bool) -> None:
        tests.append({"test": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)
    base = ["<thead>", "<tr>", '<td rowspan="2"></td>', "<td></td>", "</tr>", "<tr>", "<td></td>", "</tr>", "</thead>", "<tbody>", "<tr>", "<td></td>", "<td></td>", "</tr>", "</tbody>"]
    same = score_row("test", {"filename": "same", "pred_tokens": base, "gt_tokens": base})
    check("identical_html_scores_one", same["hh_teds"] == 1 and same["hspan_f1"] == 1)
    body_bad = base[:-4] + ["<tr>", '<td colspan="2"></td>', "</tr>", "</tbody>"]
    body = score_row("test", {"filename": "body", "pred_tokens": body_bad, "gt_tokens": base})
    check("tbody_only_change_ignored", body["hh_teds"] == 1 and body["hspan_f1"] == 1)
    span_bad = [x.replace('rowspan="2"', 'rowspan="3"') for x in base]
    span = score_row("test", {"filename": "span", "pred_tokens": span_bad, "gt_tokens": base})
    check("header_span_damage_penalized", span["hspan_f1"] < 1)
    check("cross_boundary_span_not_clipped", span["header_span_fp"] == 1 and span["header_span_fn"] == 1)
    empty = ["<thead>", "<tr>", "<td></td>", "</tr>", "</thead>"]
    empty_score = score_row("test", {"filename": "empty", "pred_tokens": empty, "gt_tokens": empty})
    check("empty_span_convention", empty_score["hspan_f1"] == 1)
    missing = score_row("test", {"filename": "missing", "pred_tokens": empty, "gt_tokens": ["<tbody>", "<tr>", "<td></td>", "</tr>", "</tbody>"]})
    check("missing_thead_is_explicit_failure", missing["parse_status"] == "missing_gt_thead")
    malformed = score_row("test", {"filename": "malformed", "pred_tokens": [], "gt_tokens": empty})
    check("malformed_prediction_is_explicit_failure", malformed["parse_status"] == "malformed_pred_rows")
    return tests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--cta-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    unit_tests = run_unit_tests()
    (args.output / "header_metric_unit_tests.json").write_text(json.dumps(unit_tests, indent=2) + "\n")
    cta_names = {row["filename"] for row in read_jsonl(args.cta_jsonl)}
    rows_by_model: dict[str, list[dict]] = {}
    summaries = []
    combined = []
    for model in MODEL_IDS:
        prediction_file = args.prediction_root / model / "sample_teds.jsonl"
        raw = [r for r in read_jsonl(prediction_file) if r["filename"] in cta_names]
        raw.sort(key=lambda r: r["filename"])
        if len(raw) != 500:
            raise RuntimeError(f"{model}: expected 500 CTA rows, got {len(raw)}")
        rows = [score_row(model, row) for row in raw]
        rows_by_model[model] = rows
        combined.extend(rows)
        write_csv(args.output / "per_model" / f"{model}.csv", rows)
        summaries.append(summarize(model, rows, prediction_file))
    write_csv(args.output / "header_metrics_per_sample.csv", combined)
    write_csv(args.output / "header_metrics_summary.csv", summaries)
    boot = bootstrap(rows_by_model, args.bootstrap, args.seed)
    write_csv(args.output / "header_bootstrap_10000.csv", boot)
    h_path = args.prediction_root / "true_horizontal" / "sample_teds.jsonl"
    v_path = args.prediction_root / "true_vertical" / "sample_teds.jsonl"
    h_rows, v_rows = rows_by_model["true_horizontal"], rows_by_model["true_vertical"]
    hv = {
        "horizontal_file": str(h_path),
        "vertical_file": str(v_path),
        "horizontal_file_sha256": hashlib.sha256(h_path.read_bytes()).hexdigest(),
        "vertical_file_sha256": hashlib.sha256(v_path.read_bytes()).hexdigest(),
        "files_identical": h_path.read_bytes() == v_path.read_bytes(),
        "cta_samples": 500,
        "different_prediction_sha256_samples": sum(a["prediction_sha256"] != b["prediction_sha256"] for a, b in zip(h_rows, v_rows)),
        "different_hh_teds_samples": sum(a["hh_teds"] != b["hh_teds"] for a, b in zip(h_rows, v_rows)),
        "different_hspan_f1_samples": sum(a["hspan_f1"] != b["hspan_f1"] for a, b in zip(h_rows, v_rows)),
    }
    (args.output / "horizontal_vertical_audit.json").write_text(json.dumps(hv, indent=2) + "\n")
    manifest = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(args.output)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_csv(args.output / "artifact_manifest.csv", manifest)
    print(json.dumps({"models": len(MODEL_IDS), "rows": len(combined), "summary": summaries, "hv": hv}, indent=2))


if __name__ == "__main__":
    main()
