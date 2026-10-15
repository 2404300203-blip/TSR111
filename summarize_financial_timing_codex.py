import csv
import glob
import json
import os
import re
import statistics


ROOT = "/root/autodl-tmp/PaddleOCR/output/deployment_export_20260812/financial"
PATTERN = re.compile(r"(.+)_(slanet|deploy)_bs(1|4)_run([123])\.json$")


def main():
    groups = {}
    for path in glob.glob(os.path.join(ROOT, "timing", "*.json")):
        match = PATTERN.match(os.path.basename(path))
        if not match:
            continue
        model_id, family, batch_size, run = match.groups()
        with open(path, "r", encoding="utf-8") as stream:
            row = json.load(stream)
        groups.setdefault((model_id, family, int(batch_size)), []).append(row)
    rows = []
    for (model_id, family, batch_size), runs in sorted(groups.items()):
        if len(runs) != 3:
            raise RuntimeError("Expected three runs for %r, got %d" % ((model_id, family, batch_size), len(runs)))
        values = lambda key: [float(run[key]) for run in runs]
        row = {
            "id": model_id,
            "model": family,
            "batch_size": batch_size,
            "runs": len(runs),
            "forward_fps_mean": statistics.fmean(values("forward_fps")),
            "forward_fps_sample_sd": statistics.stdev(values("forward_fps")),
            "pipeline_fps_mean": statistics.fmean(values("pipeline_fps")),
            "pipeline_fps_sample_sd": statistics.stdev(values("pipeline_fps")),
            "forward_latency_mean_ms": statistics.fmean(
                float(run["forward_batch_latency"]["mean_ms"]) for run in runs
            ),
            "forward_latency_p50_ms": statistics.fmean(
                float(run["forward_batch_latency"]["p50_ms"]) for run in runs
            ),
            "forward_latency_p95_ms": statistics.fmean(
                float(run["forward_batch_latency"]["p95_ms"]) for run in runs
            ),
            "peak_allocated_mib": max(run["peak_allocated_bytes"] for run in runs) / 1048576.0,
            "peak_reserved_mib": max(run["peak_reserved_bytes"] for run in runs) / 1048576.0,
        }
        rows.append(row)
    csv_path = os.path.join(ROOT, "financial_timing_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(ROOT, "financial_timing_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
