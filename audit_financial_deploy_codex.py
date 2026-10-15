import hashlib
import json
import os
import statistics


ROOT = "/root/autodl-tmp/PaddleOCR"
OUT = os.path.join(ROOT, "output/deployment_export_20260812/financial")
IDS = (
    "pilot_heldout75",
    "train2000_cached500",
    "train2000_company100",
    "company_seed20260806",
    "company_seed20260807",
    "company_seed20260808",
)


def load(path):
    with open(path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def key(row):
    return os.path.basename(row.get("filename") or row.get("image_path"))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(rows):
    return {
        "samples": len(rows),
        "teds": statistics.fmean(float(row["teds"]) for row in rows),
        "structure_acc": statistics.fmean(bool(row["exact"]) for row in rows),
    }


def main():
    results = []
    total = 0
    total_tokens = 0
    for model_id in IDS:
        full_path = os.path.join(OUT, "predictions", model_id + "_full.jsonl")
        deploy_path = os.path.join(OUT, "predictions", model_id + "_deploy.jsonl")
        full_rows = load(full_path)
        deploy_rows = load(deploy_path)
        full = {key(row): row for row in full_rows}
        deploy = {key(row): row for row in deploy_rows}
        names = sorted(set(full) | set(deploy))
        missing_full = sorted(set(deploy) - set(full))
        missing_deploy = sorted(set(full) - set(deploy))
        token_mismatches = []
        confidence_mismatches = []
        max_confidence_diff = 0.0
        for name in names:
            if name not in full or name not in deploy:
                continue
            if full[name]["pred_tokens"] != deploy[name]["pred_tokens"]:
                token_mismatches.append(name)
            diff = abs(float(full[name]["confidence"]) - float(deploy[name]["confidence"]))
            max_confidence_diff = max(max_confidence_diff, diff)
            if diff != 0.0:
                confidence_mismatches.append(name)
        compared = len(names) - len(set(missing_full) | set(missing_deploy))
        total += compared
        total_tokens += compared - len(token_mismatches)
        results.append(
            {
                "id": model_id,
                "samples": compared,
                "token_matches": compared - len(token_mismatches),
                "confidence_matches": compared - len(confidence_mismatches),
                "max_confidence_abs_diff": max_confidence_diff,
                "missing_full": missing_full,
                "missing_deploy": missing_deploy,
                "token_mismatches": token_mismatches,
                "confidence_mismatches": confidence_mismatches,
                "full_metrics": aggregate(full_rows),
                "deploy_metrics": aggregate(deploy_rows),
                "full_sha256": sha256(full_path),
                "deploy_sha256": sha256(deploy_path),
            }
        )
    output = {
        "results": results,
        "total_compared": total,
        "total_token_matches": total_tokens,
        "all_identical": total == total_tokens,
    }
    with open(os.path.join(OUT, "financial_identity_audit.json"), "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
