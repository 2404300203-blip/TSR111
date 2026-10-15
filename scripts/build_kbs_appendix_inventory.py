#!/usr/bin/env python3
"""Classify historical DBM-SLANet logs and build appendix traceability tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "remote_results/kbs_appendix_audit_20260803/remote_experiment_log_inventory.csv"
OUT = ROOT / "论文材料/kbs_appendix_experiments_20260803"
TABLES = OUT / "tables"


MAIN_OVERLAP = (
    "cta500_phase5e_eval", "cta500_baseline_eval", "cta500_expC_horizontal",
    "cta500_expD_vertical", "cta500_phase11", "phase11e_", "phase11f_",
    "complex_structure_special_eval", "phase10_axis_ablation_val5000",
    "phase10_more_models_val5000", "fintabnet_c_company_disjoint_r2",
    "fintabnet_c_doduytoan_train2000_r2",
)

CATEGORIES = (
    ("span_calibration_search", ("phase8_span_calibration",)),
    ("candidate_reranking_search", ("phase9_candidate", "phase9b_", "phase9c_", "phase9d_")),
    ("validation_scale_control", ("phase10_large_val", "statistical_validation")),
    ("progressive_training", ("E4_", "E5_", "E6", "E7", "phase6_teds", "phase7_span")),
    ("axis_ablation", ("phase10_axis", "cta500_expC", "cta500_expD")),
    ("controlled_ablation", ("phase11", "cta500_phase11", "phase10_more_models")),
    ("financial_domain_adaptation", ("fintabnet",)),
    ("cta500_primary", ("cta500_", "complex_structure_special_eval")),
)


def classify_path(path: str) -> tuple[str, str, str]:
    if any(token in path for token in MAIN_OVERLAP):
        overlap = "main_text_overlap"
    else:
        overlap = "not_in_main_text"
    category = "other"
    for name, tokens in CATEGORIES:
        if any(token in path for token in tokens):
            category = name
            break
    if "cell_presence" in path:
        validity = "excluded_label_leakage_risk"
    elif "smoke" in path or "debug" in path:
        validity = "development_only"
    else:
        validity = "eligible_after_source_check"
    return category, overlap, validity


def read_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    source_rows = read_rows()
    classified = []
    for row in source_rows:
        category, overlap, validity = classify_path(row["log_path"])
        item = dict(row)
        item.update(category=category, overlap_with_main=overlap, validity=validity)
        if row["status"] == "failed_or_interrupted":
            item["validity"] = "failed_or_interrupted"
        classified.append(item)
    fields = list(source_rows[0]) + ["category", "overlap_with_main", "validity"]
    write_csv(TABLES / "H01_historical_log_classification.csv", classified, fields)

    summary = {}
    for row in classified:
        key = (row["category"], row["overlap_with_main"], row["status"], row["validity"])
        summary[key] = summary.get(key, 0) + 1
    summary_rows = [
        {"category": key[0], "overlap_with_main": key[1], "status": key[2],
         "validity": key[3], "logs": count}
        for key, count in sorted(summary.items())
    ]
    write_csv(
        TABLES / "H02_historical_log_classification_summary.csv", summary_rows,
        ["category", "overlap_with_main", "status", "validity", "logs"],
    )

    search_rows = []
    for row in classified:
        if row["category"] not in {"span_calibration_search", "candidate_reranking_search"}:
            continue
        if not row["teds"]:
            continue
        search_rows.append({
            "search_family": row["category"], "run": Path(row["log_path"]).stem,
            "teds": row["teds"], "structure_acc": row["structure_acc"], "samples": row["samples"],
            "fps": row["fps"], "source_log": row["log_path"],
            "interpretation": "no_change_or_degradation; report as negative search",
        })
    write_csv(
        TABLES / "H03_historical_calibration_and_reranking_search.csv", search_rows,
        ["search_family", "run", "teds", "structure_acc", "samples", "fps", "source_log", "interpretation"],
    )

    trajectory_rows = []
    for row in classified:
        if row["category"] != "progressive_training" or not row["teds"]:
            continue
        if "/eval/" not in row["log_path"] and "diagnostics" not in row["log_path"]:
            continue
        trajectory_rows.append({
            "run": row["log_path"].split("/")[1], "model_head": row["model_head"],
            "teds": row["teds"], "structure_acc": row["structure_acc"], "samples": row["samples"],
            "fps": row["fps"], "source_log": row["log_path"],
        })
    write_csv(
        TABLES / "H04_progressive_training_trajectory.csv", trajectory_rows,
        ["run", "model_head", "teds", "structure_acc", "samples", "fps", "source_log"],
    )

    validation_rows = []
    for row in classified:
        if row["category"] != "validation_scale_control" or not row["teds"]:
            continue
        validation_rows.append({
            "run": Path(row["log_path"]).parent.name, "teds": row["teds"],
            "structure_acc": row["structure_acc"], "samples": row["samples"],
            "fps": row["fps"], "source_log": row["log_path"],
        })
    write_csv(
        TABLES / "H05_validation_scale_control.csv", validation_rows,
        ["run", "teds", "structure_acc", "samples", "fps", "source_log"],
    )

    manifest = {
        "source_logs": len(classified),
        "not_in_main_text": sum(row["overlap_with_main"] == "not_in_main_text" for row in classified),
        "main_text_overlap": sum(row["overlap_with_main"] == "main_text_overlap" for row in classified),
        "search_rows": len(search_rows),
        "trajectory_rows": len(trajectory_rows),
        "validation_scale_rows": len(validation_rows),
    }
    (OUT / "historical_inventory_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
