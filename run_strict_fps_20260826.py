#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import yaml


ROOT = Path("/root/autodl-tmp/PaddleOCR")
PYTHON = Path("/root/miniconda3/envs/slanet/bin/python")
OUT = ROOT / "output/deployment_header_audit_20260826"
BENCHMARK = ROOT / "output/deployment_export_20260812/benchmark_cycle_codex.py"
PUB5000 = ROOT / "train_data/table/pubtabnet/phase10_val_5000.jsonl"
CTA500 = ROOT / "train_data/table/pubtabnet/complex_structure_cta500_20260625.jsonl"

MODEL_CONFIGS = {
    "slanet": ROOT / "output/main_tables_rebuild_20260810/configs/slanet.yml",
    "cell_token": ROOT / "output/main_tables_rebuild_20260810/configs/cell_token.yml",
    "lite_mamba": ROOT / "output/main_tables_rebuild_20260810/configs/lite_mamba.yml",
    "full_mamba_no_presence": ROOT / "output/main_tables_rebuild_20260810/configs/full_mamba_no_presence.yml",
    "full_mamba_presence": ROOT / "output/main_tables_rebuild_20260810/configs/full_mamba_presence.yml",
    "full_mamba_one_way": ROOT / "output/main_tables_rebuild_20260810/configs/full_mamba_one_way.yml",
    "dbm_d64": ROOT / "output/main_tables_rebuild_20260810/configs/dbm_d64.yml",
    "dbm_ds8": ROOT / "output/main_tables_rebuild_20260810/configs/dbm_ds8.yml",
    "dbm_d192": ROOT / "output/main_tables_rebuild_20260810/configs/dbm_d192.yml",
    "dbm_e1": ROOT / "output/main_tables_rebuild_20260810/configs/dbm_e1.yml",
    "zero_geometry": ROOT / "output/main_tables_rebuild_20260810/configs/zero_geometry.yml",
    "true_horizontal": ROOT / "output/main_tables_rebuild_20260810/configs/true_horizontal.yml",
    "true_vertical": ROOT / "output/main_tables_rebuild_20260810/configs/true_vertical.yml",
    "dbm_slanet_deploy": ROOT / "output/deployment_export_20260812/dbm_slanet_deploy.yml",
}


def write_config(source: Path, output: Path, labels: Path, batch_size: int) -> None:
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg["Eval"]["dataset"]["data_dir"] = "train_data/table/pubtabnet/val/"
    cfg["Eval"]["dataset"]["label_file_list"] = [str(labels.relative_to(ROOT))]
    cfg["Eval"]["loader"]["batch_size_per_card"] = batch_size
    cfg["Eval"]["loader"]["num_workers"] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def valid_result(path: Path, batch_size: int, measured: int) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
        return row["measured_batches"] == measured and row["warmup_batches"] == 20 and row["batch_size_min"] == batch_size and row["batch_size_max"] == batch_size
    except Exception:
        return False


def run_one(dataset: str, model: str, config: Path, batch_size: int, run: int) -> Path:
    measured = 400 if batch_size == 1 else 200
    result = OUT / "strict_fps" / dataset / f"{model}.bs{batch_size}.run{run}.json"
    log = result.with_suffix(".log")
    if valid_result(result, batch_size, measured):
        print(f"cached {dataset} {model} bs{batch_size} run{run}", flush=True)
        return result
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        "FPS_BENCH_WARMUP": "20",
        "FPS_BENCH_BATCHES": str(measured),
        "FPS_BENCH_OUTPUT": str(result),
        "FPS_BENCH_RUN": str(run),
        "FLAGS_allocator_strategy": "auto_growth",
    })
    result.parent.mkdir(parents=True, exist_ok=True)
    print(f"running {dataset} {model} bs{batch_size} run{run}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run([str(PYTHON), str(BENCHMARK), "-c", str(config)], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode or not valid_result(result, batch_size, measured):
        raise RuntimeError(f"failed: {dataset} {model} bs{batch_size} run{run}; see {log}")
    return result


def aggregate(dataset: str, paths: list[Path]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for path in paths:
        name = path.name
        model, rest = name.split(".bs", 1)
        batch_size = int(rest.split(".", 1)[0])
        grouped.setdefault((model, batch_size), []).append(json.loads(path.read_text()))
    rows = []
    for (model, batch_size), runs in grouped.items():
        runs.sort(key=lambda r: int(r["run_id"]))
        row = {
            "dataset": dataset,
            "model": model,
            "batch_size": batch_size,
            "independent_processes": len(runs),
            "warmup_batches": runs[0]["warmup_batches"],
            "measured_batches_per_process": runs[0]["measured_batches"],
        }
        for field in ("forward_fps", "pipeline_fps"):
            values = [float(r[field]) for r in runs]
            row[field + "_mean"] = statistics.fmean(values)
            row[field + "_sample_sd"] = statistics.stdev(values)
        for prefix in ("forward_batch_latency", "pipeline_batch_latency"):
            for metric in ("mean_ms", "p50_ms", "p95_ms"):
                row[prefix + "_" + metric + "_mean"] = statistics.fmean(float(r[prefix][metric]) for r in runs)
        row["peak_allocated_mib_max"] = max(r["peak_allocated_bytes"] for r in runs) / 2**20
        row["peak_reserved_mib_max"] = max(r["peak_reserved_bytes"] for r in runs) / 2**20
        row["checkpoint"] = runs[0]["checkpoint"]
        row["config_sha256"] = hashlib.sha256(Path(runs[0]["config"]).read_bytes()).hexdigest()
        rows.append(row)
    return sorted(rows, key=lambda r: (r["model"], r["batch_size"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_pub_cta() -> None:
    generated = OUT / "strict_fps" / "configs"
    for dataset, labels in (("pubtabnet5000", PUB5000), ("cta500", CTA500)):
        paths = []
        for model, source in MODEL_CONFIGS.items():
            for batch_size in (1, 4):
                config = generated / dataset / f"{model}.bs{batch_size}.yml"
                write_config(source, config, labels, batch_size)
                for run in (1, 2, 3):
                    paths.append(run_one(dataset, model, config, batch_size, run))
        write_csv(OUT / f"strict_fps_{dataset}.csv", aggregate(dataset, paths))


def run_financial() -> None:
    manifest = json.loads((ROOT / "output/deployment_export_20260812/financial/timing_manifest.json").read_text())
    paths = []
    for item in manifest:
        model = f"{item['id']}__{item['family']}"
        batch_size = int(item["batch_size"])
        source = Path(item["config"])
        cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
        cfg["Eval"]["loader"]["batch_size_per_card"] = batch_size
        cfg["Eval"]["loader"]["num_workers"] = 0
        cfg["Eval"]["loader"]["drop_last"] = batch_size > 1
        config = OUT / "strict_fps" / "configs" / "financial" / f"{model}.bs{batch_size}.yml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        for run in (1, 2, 3):
            paths.append(run_one("financial", model, config, batch_size, run))
    write_csv(OUT / "strict_fps_financial.csv", aggregate("financial", paths))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_pub_cta()
    run_financial()
    (OUT / "strict_fps.COMPLETE").touch()


if __name__ == "__main__":
    main()
