#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for CTA-500 header metrics."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "remote_results" / "header_metrics_20260814" / "cta500_header_metrics_per_sample.csv"
OUTPUT = ROOT / "remote_results" / "header_metrics_20260814" / "cta500_header_metrics_bootstrap.csv"
BASELINES = ["slanet", "cell_token", "true_horizontal", "true_vertical"]
LABELS = {
    "slanet": "SLANet",
    "cell_token": "Cell Token only",
    "true_horizontal": "True Horizontal-only",
    "true_vertical": "True Vertical-only",
    "dbm_slanet": "DBM-SLANet",
}


def quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def metrics(rows: list[dict], indices: list[int]) -> tuple[float, float]:
    hh_teds = sum(float(rows[i]["hh_teds"]) for i in indices) / len(indices)
    tp = sum(int(rows[i]["span_tp"]) for i in indices)
    fp = sum(int(rows[i]["span_fp"]) for i in indices)
    fn = sum(int(rows[i]["span_fn"]) for i in indices)
    hspan = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return hh_teds, hspan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["group"], row["model_id"])].append(row)

    output = []
    rng = random.Random(args.seed)
    for group in sorted({key[0] for key in grouped}):
        dbm = sorted(grouped[(group, "dbm_slanet")], key=lambda row: row["filename"])
        filenames = [row["filename"] for row in dbm]
        n = len(dbm)
        full_indices = list(range(n))
        for baseline_id in BASELINES:
            baseline = sorted(grouped[(group, baseline_id)], key=lambda row: row["filename"])
            if [row["filename"] for row in baseline] != filenames:
                raise RuntimeError(f"Unpaired filenames for {group}/{baseline_id}")
            dbm_point = metrics(dbm, full_indices)
            baseline_point = metrics(baseline, full_indices)
            point = (dbm_point[0] - baseline_point[0], dbm_point[1] - baseline_point[1])
            hh_samples, span_samples = [], []
            for _ in range(args.iterations):
                sample = [rng.randrange(n) for _ in range(n)]
                dbm_score = metrics(dbm, sample)
                baseline_score = metrics(baseline, sample)
                hh_samples.append(dbm_score[0] - baseline_score[0])
                span_samples.append(dbm_score[1] - baseline_score[1])
            output.append({
                "group": group,
                "n": n,
                "comparison": f"DBM-SLANet minus {LABELS[baseline_id]}",
                "hh_teds_delta": f"{point[0]:.12f}",
                "hh_teds_ci_low": f"{quantile(hh_samples, 0.025):.12f}",
                "hh_teds_ci_high": f"{quantile(hh_samples, 0.975):.12f}",
                "hspan_f1_delta": f"{point[1]:.12f}",
                "hspan_f1_ci_low": f"{quantile(span_samples, 0.025):.12f}",
                "hspan_f1_ci_high": f"{quantile(span_samples, 0.975):.12f}",
                "iterations": args.iterations,
                "seed": args.seed,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} paired comparisons to {args.output}")


if __name__ == "__main__":
    main()
