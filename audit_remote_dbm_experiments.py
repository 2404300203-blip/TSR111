#!/usr/bin/env python3
"""Build a traceable inventory from DBM-SLANet remote experiment logs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent if (SCRIPT.parent / "output").is_dir() else SCRIPT.parents[1]
OUTPUT_ROOT = ROOT / "output"
DEST = OUTPUT_ROOT / "kbs_appendix_audit_20260803"


def last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def last_text(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1].strip() if matches else ""


def inspect_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    status = "completed" if any(
        marker in lower
        for marker in ("structure teds diagnostics eval", "metric eval", "best metric")
    ) else "incomplete"
    if "traceback (most recent call last)" in lower or "error:" in lower:
        status = "failed_or_interrupted"

    dataset = last_text(r"label_file_list\s*:\s*\['([^']+)'\]", text)
    checkpoint = last_text(r"checkpoints\s*:\s*(.+)$", text)
    model_head = last_text(r"^.*?name\s*:\s*(Cell\w*(?:Mamba|Token)\w*Head|SLAHead)\s*$", text)
    teds = last_float(r"(?:^|\s)teds\s*:\s*([0-9.]+)", text)
    structure_acc = last_float(r"structure_acc\s*:\s*([0-9.]+)", text)
    samples = last_float(r"(?:^|\s)samples\s*:\s*([0-9.]+)", text)
    fps = last_float(r"(?:^|\s)fps\s*:\s*([0-9.]+)", text)
    eval_acc = last_float(r"(?:^|[, {])acc\s*[:=]\s*([0-9.]+)", text)
    best_acc = last_float(r"best.*?acc['\"]?\s*[:=]\s*([0-9.]+)", text)

    return {
        "log_path": str(path.relative_to(ROOT)),
        "status": status,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "model_head": model_head,
        "teds": teds,
        "structure_acc": structure_acc,
        "samples": int(samples) if samples is not None else None,
        "fps": fps,
        "eval_acc": eval_acc,
        "best_acc": best_acc,
        "bytes": path.stat().st_size,
    }


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    logs = sorted(OUTPUT_ROOT.glob("**/*.log"))
    rows = [inspect_log(path) for path in logs]

    csv_path = DEST / "remote_experiment_log_inventory.csv"
    fieldnames = [
        "log_path", "status", "dataset", "checkpoint", "model_head", "teds",
        "structure_acc", "samples", "fps", "eval_acc", "best_acc", "bytes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = DEST / "remote_experiment_log_inventory.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "logs": len(rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "failed_or_interrupted": sum(row["status"] == "failed_or_interrupted" for row in rows),
        "with_teds": sum(row["teds"] is not None for row in rows),
        "with_structure_acc": sum(row["structure_acc"] is not None for row in rows),
        "with_fps": sum(row["fps"] is not None for row in rows),
    }
    (DEST / "remote_experiment_log_inventory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
