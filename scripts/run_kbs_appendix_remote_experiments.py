#!/usr/bin/env python3
"""Run traceable, non-destructive appendix evaluation controls on the remote GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path("/root/autodl-tmp/PaddleOCR")
PYTHON = Path("/root/miniconda3/envs/slanet/bin/python")
OUT = ROOT / "output/kbs_appendix_new_experiments_20260803"
CONFIG_OUT = OUT / "configs"
RESULTS = OUT / "appendix_new_experiment_results.csv"
SOURCE_LABEL = ROOT / "train_data/table/pubtabnet/complex_structure_cta500_20260625.jsonl"
SOURCE_IMAGES = ROOT / "train_data/table/pubtabnet/val"
VARIANT_ROOT = ROOT / "train_data/table/pubtabnet/appendix_cta500_variants_20260803"

BASE_CONFIGS = {
    "SLANet": ROOT / "configs/table/SLANet_pubtabnet_baseline_eval_cta500_codex.yml",
    "DBM-SLANet": ROOT / "configs/table/DBM_SLANet_phase5e_eval_cta500_codex.yml",
}

VARIANTS = {
    "jpeg_q75": ("jpeg", 75.0),
    "jpeg_q50": ("jpeg", 50.0),
    "blur_sigma0p8": ("blur", 0.8),
    "blur_sigma1p2": ("blur", 1.2),
    "brightness0p75": ("brightness", 0.75),
    "brightness1p25": ("brightness", 1.25),
    "contrast0p75": ("contrast", 0.75),
}


def load_labels() -> list[dict]:
    with SOURCE_LABEL.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def apply_variant(image: np.ndarray, kind: str, value: float) -> np.ndarray:
    if kind == "jpeg":
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(value)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if kind == "blur":
        return cv2.GaussianBlur(image, (0, 0), sigmaX=value, sigmaY=value)
    if kind == "brightness":
        return np.clip(image.astype(np.float32) * value, 0, 255).astype(np.uint8)
    if kind == "contrast":
        mean = image.mean(axis=(0, 1), keepdims=True)
        return np.clip((image.astype(np.float32) - mean) * value + mean, 0, 255).astype(np.uint8)
    raise ValueError(kind)


def prepare_variants(labels: list[dict]) -> None:
    for name, (kind, value) in VARIANTS.items():
        target = VARIANT_ROOT / name
        target.mkdir(parents=True, exist_ok=True)
        label_path = VARIANT_ROOT / f"{name}.jsonl"
        if label_path.exists() and sum(1 for _ in label_path.open(encoding="utf-8")) == len(labels):
            continue
        for index, record in enumerate(labels, start=1):
            filename = record["filename"]
            src = SOURCE_IMAGES / filename
            dst = target / filename
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(src)
            transformed = apply_variant(image, kind, value)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dst), transformed):
                raise RuntimeError(f"Could not write {dst}")
            if index % 100 == 0:
                print(f"prepared {name}: {index}/{len(labels)}", flush=True)
        with label_path.open("w", encoding="utf-8") as handle:
            for record in labels:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def replace_resize(config: dict, size: int) -> None:
    for split in ("Train", "Eval"):
        for transform in config[split]["dataset"]["transforms"]:
            if "ResizeTableImage" in transform:
                transform["ResizeTableImage"]["max_len"] = size
            if "PaddingTableImage" in transform:
                transform["PaddingTableImage"]["size"] = [size, size]


def build_config(model: str, experiment_id: str, *, resolution: int = 488,
                 batch_size: int = 4, variant: str | None = None) -> Path:
    with BASE_CONFIGS[model].open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_dir = OUT / experiment_id
    config["Global"]["save_model_dir"] = str(run_dir)
    config["Global"]["save_res_path"] = str(run_dir / "infer")
    config["Global"]["diagnostics_path"] = str(run_dir / "sample_teds.jsonl")
    replace_resize(config, resolution)
    config["Eval"]["loader"]["batch_size_per_card"] = batch_size
    config["Eval"]["loader"]["num_workers"] = 0
    if variant:
        config["Eval"]["dataset"]["data_dir"] = str(VARIANT_ROOT / variant) + "/"
        config["Eval"]["dataset"]["label_file_list"] = [str(VARIANT_ROOT / f"{variant}.jsonl")]
    CONFIG_OUT.mkdir(parents=True, exist_ok=True)
    path = CONFIG_OUT / f"{experiment_id}.yml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def existing_ids() -> set[str]:
    if not RESULTS.exists():
        return set()
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        return {row["experiment_id"] for row in csv.DictReader(handle) if row["status"] == "completed"}


def parse_last(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def monitor_memory(process: subprocess.Popen, state: dict) -> None:
    peak = 0
    while process.poll() is None:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 2 and fields[0] == str(process.pid):
                peak = max(peak, int(fields[1]))
        time.sleep(0.2)
    state["peak_memory_mib"] = peak or None


def append_result(row: dict) -> None:
    fields = [
        "experiment_id", "category", "model", "resolution", "batch_size", "variant",
        "repeat", "status", "samples", "teds", "structure_acc", "fps", "wall_time_s",
        "peak_memory_mib", "config", "log", "diagnostics", "error_note",
    ]
    write_header = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_job(job: dict) -> None:
    experiment_id = job["experiment_id"]
    config = build_config(
        job["model"], experiment_id, resolution=job.get("resolution", 488),
        batch_size=job.get("batch_size", 4), variant=job.get("variant"),
    )
    run_dir = OUT / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "teds_eval.log"
    start = time.time()
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")
    child_env["OMP_NUM_THREADS"] = "1"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [str(PYTHON), "scripts/eval_table_teds_codex.py", "-c", str(config)],
            cwd=ROOT, env=child_env, stdout=log_handle, stderr=subprocess.STDOUT,
        )
        memory = {}
        watcher = threading.Thread(target=monitor_memory, args=(process, memory), daemon=True)
        watcher.start()
        return_code = process.wait()
        watcher.join(timeout=2)
    wall = time.time() - start
    text = log_path.read_text(encoding="utf-8", errors="replace")
    completed = return_code == 0 and "structure TEDS eval" in text
    row = {
        **job,
        "resolution": job.get("resolution", 488),
        "batch_size": job.get("batch_size", 4),
        "variant": job.get("variant", "clean"),
        "repeat": job.get("repeat", ""),
        "status": "completed" if completed else "failed",
        "samples": parse_last(r"samples:([0-9.]+)", text),
        "teds": parse_last(r"teds:([0-9.]+)", text),
        "structure_acc": parse_last(r"structure_acc:([0-9.]+)", text),
        "fps": parse_last(r"fps:([0-9.]+)", text),
        "wall_time_s": round(wall, 3),
        "peak_memory_mib": memory.get("peak_memory_mib"),
        "config": str(config.relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
        "diagnostics": "",
        "error_note": "" if completed else f"teds_return_code={return_code}",
    }
    append_result(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def jobs() -> list[dict]:
    result = []
    for model in BASE_CONFIGS:
        short = "slanet" if model == "SLANet" else "dbm"
        for resolution in (384, 448, 576, 640):
            result.append({
                "experiment_id": f"resolution_{short}_{resolution}", "category": "resolution",
                "model": model, "resolution": resolution,
            })
        for batch_size in (1, 2, 8):
            result.append({
                "experiment_id": f"batch_{short}_{batch_size}", "category": "batch_size",
                "model": model, "batch_size": batch_size,
            })
        for variant in VARIANTS:
            result.append({
                "experiment_id": f"robustness_{short}_{variant}", "category": "robustness",
                "model": model, "variant": variant,
            })
        for repeat in (1, 2, 3):
            result.append({
                "experiment_id": f"timing_{short}_repeat{repeat}", "category": "timing_repeat",
                "model": model, "repeat": repeat,
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=[], help="Run only these experiment IDs")
    parser.add_argument("--skip-variants", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    if len(labels) != 500:
        raise ValueError(f"Expected CTA-500, found {len(labels)} labels")
    selected = set(args.only)
    all_jobs = [job for job in jobs() if not selected or job["experiment_id"] in selected]
    if any(job.get("variant") for job in all_jobs) and not args.skip_variants:
        prepare_variants(labels)
    done = existing_ids()
    print(f"jobs={len(all_jobs)} already_completed={len(done)}", flush=True)
    for index, job in enumerate(all_jobs, start=1):
        if job["experiment_id"] in done:
            continue
        print(f"[{index}/{len(all_jobs)}] {job['experiment_id']}", flush=True)
        run_job(job)


if __name__ == "__main__":
    main()
