import hashlib
import json
import os
import statistics


ROOT = "/root/autodl-tmp/PaddleOCR"
OUT = os.path.join(ROOT, "output/deployment_export_20260812")
FULL = os.path.join(ROOT, "output/fps_optimization_20260812/predictions/dbm_full.jsonl")
DEPLOY = os.path.join(OUT, "predictions/pubtabnet5000_deploy.jsonl")
SUBSETS = {
    "pubtabnet5000": None,
    "complex4086": os.path.join(
        ROOT, "train_data/table/pubtabnet/complex_structure_special_full_4086_20260625.jsonl"
    ),
    "cta500": os.path.join(
        ROOT, "train_data/table/pubtabnet/complex_structure_cta500_20260625.jsonl"
    ),
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def filename(record):
    value = record.get("filename") or record.get("image") or record.get("image_path")
    if not value:
        raise KeyError("No filename in record: %r" % record)
    return os.path.basename(value)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_subset(name, names, full, deploy):
    selected = sorted(names if names is not None else set(full))
    missing_full = sorted(set(selected) - set(full))
    missing_deploy = sorted(set(selected) - set(deploy))
    token_mismatches = []
    confidence_mismatches = []
    teds_mismatches = []
    max_confidence_diff = 0.0
    for item in selected:
        if item not in full or item not in deploy:
            continue
        left, right = full[item], deploy[item]
        if left["pred_tokens"] != right["pred_tokens"]:
            token_mismatches.append(item)
        confidence_diff = abs(float(left["confidence"]) - float(right["confidence"]))
        max_confidence_diff = max(max_confidence_diff, confidence_diff)
        if confidence_diff != 0.0:
            confidence_mismatches.append(item)
        if abs(float(left["teds"]) - float(right["teds"])) != 0.0:
            teds_mismatches.append(item)
    compared = len(selected) - len(set(missing_full) | set(missing_deploy))
    return {
        "subset": name,
        "expected_samples": len(selected),
        "compared_samples": compared,
        "token_matches": compared - len(token_mismatches),
        "confidence_matches": compared - len(confidence_mismatches),
        "teds_matches": compared - len(teds_mismatches),
        "max_confidence_abs_diff": max_confidence_diff,
        "missing_full": missing_full,
        "missing_deploy": missing_deploy,
        "token_mismatches": token_mismatches,
        "confidence_mismatches": confidence_mismatches,
        "teds_mismatches": teds_mismatches,
    }


def timing_summary():
    result = {}
    for bs in (1, 4):
        rows = []
        for run in (1, 2, 3):
            path = os.path.join(OUT, "timing", "deploy_bs%d_run%d.json" % (bs, run))
            with open(path, "r", encoding="utf-8") as stream:
                rows.append(json.load(stream))
        fields = ("forward_fps", "pipeline_fps")
        summary = {"batch_size": bs, "runs": rows}
        for field in fields:
            values = [float(row[field]) for row in rows]
            summary[field + "_mean"] = statistics.fmean(values)
            summary[field + "_sample_sd"] = statistics.stdev(values)
        for family in ("forward_batch_latency", "pipeline_batch_latency"):
            for stat in ("mean_ms", "p50_ms", "p95_ms"):
                values = [float(row[family][stat]) for row in rows]
                summary[family + "_" + stat + "_mean"] = statistics.fmean(values)
        summary["peak_allocated_bytes_max"] = max(row["peak_allocated_bytes"] for row in rows)
        summary["peak_reserved_bytes_max"] = max(row["peak_reserved_bytes"] for row in rows)
        result["bs%d" % bs] = summary
    return result


def main():
    full_rows = load_jsonl(FULL)
    deploy_rows = load_jsonl(DEPLOY)
    full = {filename(row): row for row in full_rows}
    deploy = {filename(row): row for row in deploy_rows}
    audits = {}
    for name, subset_path in SUBSETS.items():
        names = None
        if subset_path:
            names = {filename(row) for row in load_jsonl(subset_path)}
        audits[name] = audit_subset(name, names, full, deploy)
    result = {
        "full_predictions": FULL,
        "deploy_predictions": DEPLOY,
        "full_prediction_sha256": sha256(FULL),
        "deploy_prediction_sha256": sha256(DEPLOY),
        "full_rows": len(full_rows),
        "deploy_rows": len(deploy_rows),
        "audits": audits,
        "timing": timing_summary(),
    }
    path = os.path.join(OUT, "physical_deploy_audit.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
