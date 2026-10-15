import csv
import glob
import json
import os
import statistics


ROOT = "/root/autodl-tmp/PaddleOCR/output/deployment_export_20260812/cta_all_models"


def main():
    groups = {}
    for path in glob.glob(os.path.join(ROOT, "*_run*.json")):
        name = os.path.basename(path).rsplit("_run", 1)[0]
        with open(path, "r", encoding="utf-8") as stream:
            groups.setdefault(name, []).append(json.load(stream))
    rows = []
    for name, runs in sorted(groups.items()):
        if len(runs) != 3:
            raise RuntimeError((name, len(runs)))
        rows.append({
            "model": name,
            "runs": 3,
            "forward_fps_mean": statistics.fmean(r["forward_fps"] for r in runs),
            "forward_fps_sample_sd": statistics.stdev(r["forward_fps"] for r in runs),
            "pipeline_fps_mean": statistics.fmean(r["pipeline_fps"] for r in runs),
            "pipeline_fps_sample_sd": statistics.stdev(r["pipeline_fps"] for r in runs),
            "peak_allocated_mib": max(r["peak_allocated_bytes"] for r in runs) / 1048576.0,
        })
    with open(os.path.join(ROOT, "summary.csv"), "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
