#!/usr/bin/env python3
"""Measure process-level peak GPU memory for every model plotted in Fig. 1."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import yaml


ROOT = Path("/root/autodl-tmp/PaddleOCR")
OUT = Path("/root/autodl-tmp/external_eval_20260811/fig1_memory_profile")
INTERNAL_PYTHON = Path("/root/miniconda3/envs/slanet/bin/python")
EXTERNAL_PYTHON = Path("/root/autodl-tmp/venvs/external_tsr/bin/python")
EXTERNAL_MODELS = Path("/root/autodl-tmp/external_models")
LABELS = ROOT / "train_data/table/pubtabnet/complex_structure_cta500_20260625.jsonl"
IMAGE_DIR = ROOT / "train_data/table/pubtabnet/val"

INTERNAL_MODELS = [
    ("slanet", "SLANet"),
    ("cell_token", "Cell Token only"),
    ("lite_mamba", "Lite Mamba"),
    ("full_mamba_no_presence", "Full Mamba, no presence"),
    ("full_mamba_presence", "Full Mamba, presence"),
    ("full_mamba_one_way", "Full Mamba, one-way"),
    ("dbm_d64", "Full DBM, d=64"),
    ("dbm_ds8", "Full DBM, d_s=8"),
    ("dbm_d192", "Full DBM, d=192"),
    ("dbm_e1", "Full DBM, e=1"),
    ("zero_geometry", "Full Mamba, zero geometry"),
    ("true_horizontal", "True Horizontal-only"),
    ("true_vertical", "True Vertical-only"),
    ("dbm_slanet", "DBM-SLANet"),
]


def monitor_memory(process: subprocess.Popen, state: dict) -> None:
    peak = 0
    samples = 0
    while process.poll() is None:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        for line in query.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 2 and fields[0] == str(process.pid):
                peak = max(peak, int(fields[1]))
                samples += 1
        time.sleep(0.02)
    state["peak_memory_mib"] = peak or None
    state["memory_samples"] = samples


def run_measured(model_id: str, model_name: str, family: str, command: list[str], cwd: Path) -> dict:
    run_dir = OUT / model_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        memory: dict = {}
        watcher = threading.Thread(target=monitor_memory, args=(process, memory), daemon=True)
        watcher.start()
        return_code = process.wait()
        watcher.join(timeout=3)
    row = {
        "model_id": model_id,
        "model": model_name,
        "family": family,
        "status": "completed" if return_code == 0 and memory.get("peak_memory_mib") else "failed",
        "peak_gpu_memory_mib": memory.get("peak_memory_mib"),
        "memory_samples": memory.get("memory_samples", 0),
        "samples": 10,
        "batch_size": 1,
        "wall_time_s": round(time.time() - start, 3),
        "return_code": return_code,
        "log": str(log_path),
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    time.sleep(0.5)
    return row


def prepare_internal_config(model_id: str, cta10: Path) -> Path:
    source = ROOT / "output/main_tables_rebuild_20260810/configs" / f"{model_id}.yml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    run_dir = OUT / model_id
    config["Global"]["save_model_dir"] = str(run_dir)
    config["Global"]["save_res_path"] = str(run_dir / "infer")
    config["Global"]["diagnostics_path"] = str(run_dir / "diagnostics.jsonl")
    config["Eval"]["dataset"]["data_dir"] = str(IMAGE_DIR) + "/"
    config["Eval"]["dataset"]["label_file_list"] = [str(cta10)]
    config["Eval"]["loader"]["batch_size_per_card"] = 1
    config["Eval"]["loader"]["num_workers"] = 0
    config_path = run_dir / "config.yml"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cta10 = OUT / "cta10.jsonl"
    with LABELS.open("r", encoding="utf-8") as source, cta10.open("w", encoding="utf-8") as target:
        for index, line in enumerate(source):
            if index == 10:
                break
            target.write(line)

    rows = []
    for model_id, model_name in INTERNAL_MODELS:
        config_path = prepare_internal_config(model_id, cta10)
        rows.append(
            run_measured(
                model_id,
                model_name,
                "internal",
                [
                    str(INTERNAL_PYTHON),
                    "eval_table_teds_codex.py",
                    "-c",
                    str(config_path),
                ],
                ROOT,
            )
        )

    rows.append(
        run_measured(
            "unitable_large",
            "UniTable Large",
            "zero-shot external baseline",
            [
                str(EXTERNAL_PYTHON),
                str(EXTERNAL_MODELS / "eval_unitable_cta500.py"),
                "--unitable-root",
                str(EXTERNAL_MODELS / "unitable"),
                "--checkpoint",
                "/root/autodl-tmp/external_weights_20260811/unitable_large_structure.pt",
                "--labels",
                str(LABELS),
                "--image-dir",
                str(IMAGE_DIR),
                "--output-dir",
                str(OUT / "unitable_large" / "predictions"),
                "--warmup",
                "0",
                "--limit",
                "10",
            ],
            EXTERNAL_MODELS / "unitable",
        )
    )
    rows.append(
        run_measured(
            "table_transformer_v1",
            "Table Transformer v1.0",
            "zero-shot external baseline",
            [
                str(EXTERNAL_PYTHON),
                str(EXTERNAL_MODELS / "eval_table_transformer_cta500.py"),
                "--microsoft-root",
                str(EXTERNAL_MODELS / "table_transformer"),
                "--checkpoint",
                "/root/autodl-tmp/external_weights_20260811/table_transformer_structure_pytorch_model.bin",
                "--labels",
                str(LABELS),
                "--image-dir",
                str(IMAGE_DIR),
                "--output-dir",
                str(OUT / "table_transformer_v1" / "predictions"),
                "--warmup",
                "0",
                "--limit",
                "10",
            ],
            EXTERNAL_MODELS,
        )
    )

    fields = list(rows[0])
    with (OUT / "fig1_peak_gpu_memory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "fig1_peak_gpu_memory.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
