import csv
import glob
import hashlib
import json
import os
import re
import statistics

import numpy as np


ROOT = "/root/autodl-tmp/PaddleOCR"
OUT = os.path.join(ROOT, "output/fps_optimization_20260812")
SUMMARY = os.path.join(OUT, "summary")
os.makedirs(SUMMARY, exist_ok=True)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_timing():
    groups = {}
    for path in sorted(glob.glob(os.path.join(OUT, "timing/*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        filename = os.path.basename(path)
        match = re.match(r"(.+)\.bs([0-9]+)\.run([0-9]+)\.json$", filename)
        if not match:
            continue
        model_id, batch_size, _ = match.groups()
        groups.setdefault((model_id, int(batch_size)), []).append(row)

    rows = []
    for (model_id, batch_size), records in sorted(groups.items()):
        forward_fps = [record["forward_fps"] for record in records]
        pipeline_fps = [record["pipeline_fps"] for record in records]
        forward_mean = [
            record["forward_batch_latency"]["mean_ms"] for record in records
        ]
        forward_p95 = [
            record["forward_batch_latency"]["p95_ms"] for record in records
        ]
        pipeline_mean = [
            record["pipeline_batch_latency"]["mean_ms"] for record in records
        ]
        peak_allocated = [record["peak_allocated_bytes"] for record in records]
        peak_reserved = [record["peak_reserved_bytes"] for record in records]
        rows.append(
            {
                "id": model_id,
                "batch_size": batch_size,
                "runs": len(records),
                "forward_fps_mean": statistics.fmean(forward_fps),
                "forward_fps_sd": sample_std(forward_fps),
                "pipeline_fps_mean": statistics.fmean(pipeline_fps),
                "pipeline_fps_sd": sample_std(pipeline_fps),
                "forward_latency_mean_ms": statistics.fmean(forward_mean),
                "forward_latency_p95_mean_ms": statistics.fmean(forward_p95),
                "pipeline_latency_mean_ms": statistics.fmean(pipeline_mean),
                "peak_allocated_mib_mean": statistics.fmean(peak_allocated)
                / (1024.0**2),
                "peak_reserved_mib_mean": statistics.fmean(peak_reserved)
                / (1024.0**2),
            }
        )
    write_csv(os.path.join(SUMMARY, "timing_summary.csv"), rows)
    return rows


def parse_eval_logs():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, "logs/*.eval*.log"))):
        text = open(path, "r", encoding="utf-8", errors="replace").read()
        values = {}
        for key in ("teds", "structure_acc", "samples", "fps"):
            matches = re.findall(rf"{key}:([0-9.]+)", text)
            values[key] = float(matches[-1]) if matches else None
        if values["samples"] is None:
            continue
        rows.append({"log": os.path.basename(path), **values})
    write_csv(os.path.join(SUMMARY, "eval_summary.csv"), rows)
    return rows


def load_predictions(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compare_predictions():
    pairs = [
        ("dbm", "dbm_full.jsonl", "dbm_pruned.jsonl"),
        (
            "three_scan_hbi_vuni",
            "three_scan_hbi_vuni_full_pub5000.jsonl",
            "three_scan_hbi_vuni_pruned_pub5000.jsonl",
        ),
        (
            "three_scan_huni_vbi",
            "three_scan_huni_vbi_full_pub5000.jsonl",
            "three_scan_huni_vbi_pruned_pub5000.jsonl",
        ),
        (
            "shared_4dir",
            "shared_4dir_full_pub5000.jsonl",
            "shared_4dir_pruned_pub5000.jsonl",
        ),
    ]
    comparisons = []
    for model_id, full_name, pruned_name in pairs:
        full_path = os.path.join(OUT, "predictions", full_name)
        pruned_path = os.path.join(OUT, "predictions", pruned_name)
        if not os.path.exists(full_path) or not os.path.exists(pruned_path):
            continue
        full = load_predictions(full_path)
        pruned = load_predictions(pruned_path)
        paired = min(len(full), len(pruned))
        token_matches = sum(
            full[idx]["pred_tokens"] == pruned[idx]["pred_tokens"]
            for idx in range(paired)
        )
        confidence_matches = sum(
            full[idx]["confidence"] == pruned[idx]["confidence"]
            for idx in range(paired)
        )
        comparisons.append(
            {
                "id": model_id,
                "full_samples": len(full),
                "pruned_samples": len(pruned),
                "token_matches": token_matches,
                "confidence_matches": confidence_matches,
                "token_mismatches": paired - token_matches,
            }
        )
    path = os.path.join(SUMMARY, "prediction_identity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2, sort_keys=True)
        f.write("\n")
    return comparisons


def parse_training_logs():
    rows = []
    pattern = re.compile(
        r"global_step: ([0-9]+).*?acc: ([0-9.]+), loss: ([0-9.]+), "
        r"structure_loss: ([0-9.]+), loc_loss: ([0-9.]+).*?"
        r"cell_presence_loss: ([0-9.]+).*?ips: ([0-9.]+) samples/s"
    )
    for path in sorted(glob.glob(os.path.join(OUT, "checkpoints/*/train.log"))):
        text = open(path, "r", encoding="utf-8", errors="replace").read()
        matches = pattern.findall(text)
        if not matches:
            continue
        step, acc, loss, structure, loc, presence, ips = matches[-1]
        rows.append(
            {
                "id": os.path.basename(os.path.dirname(path)),
                "last_step": int(step),
                "last_logged_acc": float(acc),
                "last_logged_loss": float(loss),
                "last_structure_loss": float(structure),
                "last_loc_loss": float(loc),
                "last_presence_loss": float(presence),
                "last_ips": float(ips),
            }
        )
    write_csv(os.path.join(SUMMARY, "training_summary.csv"), rows)
    return rows


def label_length_stats():
    path = os.path.join(
        ROOT, "train_data/table/pubtabnet/phase10_val_5000.jsonl"
    )
    lengths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                lengths.append(len(record["html"]["structure"]["tokens"]))
    result = {
        "samples": len(lengths),
        "min": min(lengths),
        "p25": float(np.percentile(lengths, 25)),
        "median": float(np.median(lengths)),
        "mean": float(np.mean(lengths)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "max_before_500_token_truncation": max(lengths),
    }
    with open(os.path.join(SUMMARY, "label_length_stats.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    return result


def write_manifest():
    paths = []
    for directory, _, filenames in os.walk(OUT):
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.basename(path) == "artifact_sha256.txt":
                continue
            paths.append(path)
    with open(os.path.join(SUMMARY, "artifact_sha256.txt"), "w") as f:
        for path in sorted(paths):
            digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
            f.write(f"{digest}  {os.path.relpath(path, OUT)}\n")


def main():
    result = {
        "timing": aggregate_timing(),
        "eval": parse_eval_logs(),
        "prediction_identity": compare_predictions(),
        "training": parse_training_logs(),
        "label_lengths": label_length_stats(),
    }
    with open(os.path.join(SUMMARY, "complete_summary.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    write_manifest()


if __name__ == "__main__":
    main()
