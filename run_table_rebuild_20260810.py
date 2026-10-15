#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

import paddle
import yaml

ROOT = Path("/root/autodl-tmp/PaddleOCR")
PYTHON = Path("/root/miniconda3/envs/slanet/bin/python")
OUT = ROOT / "output/main_tables_rebuild_20260810"
CONFIG_DIR = OUT / "configs"
LOG_DIR = OUT / "logs"
PUB5000 = ROOT / "train_data/table/pubtabnet/phase10_val_5000.jsonl"
CTA500 = ROOT / "train_data/table/pubtabnet/complex_structure_cta500_20260625.jsonl"
VAL_DIR = "train_data/table/pubtabnet/val/"

MODELS = [
    {
        "id": "slanet",
        "name": "SLANet",
        "base_config": "configs/table/SLANet_pubtabnet_baseline_eval_cta500_codex.yml",
        "checkpoint": "./output/baseline_package_20260621/model/best_accuracy",
        "table9": False,
        "table10": False,
    },
    {
        "id": "cell_token",
        "name": "Cell Token only",
        "base_config": "configs/table/DBM_SLANet_phase11a_cell_token_only_30000_from_baseline_codex.yml",
        "checkpoint": "./output/E11A_cell_token_only_30000_from_baseline_20260708/best_accuracy",
        "table9": True,
        "table10": False,
    },
    {
        "id": "lite_mamba",
        "name": "Lite Mamba",
        "base_config": "configs/table/DBM_SLANet_full_mamba_lite_stage7_30000_lr5e6_from_stage6_codex.yml",
        "checkpoint": "./output/E4_lite_stage7_lr5e6_from_stage6_ft_30000_20260623/best_accuracy",
        "table9": True,
        "table10": False,
    },
    {
        "id": "full_mamba_no_presence",
        "name": "Full Mamba, no presence",
        "base_config": "configs/table/DBM_SLANet_phase11b_full_mamba_lambda0_30000_from_baseline_codex.yml",
        "checkpoint": "./output/E11B_full_mamba_lambda0_30000_from_baseline_20260708/best_accuracy",
        "table9": True,
        "table10": False,
    },
    {
        "id": "full_mamba_presence",
        "name": "Full Mamba, presence",
        "base_config": "configs/table/DBM_SLANet_phase11c_full_mamba_lambda02_30000_from_baseline_codex.yml",
        "checkpoint": "./output/E11C_full_mamba_lambda02_30000_from_baseline_20260709/best_accuracy",
        "table9": True,
        "table10": False,
    },
    {
        "id": "full_mamba_one_way",
        "name": "Full Mamba, one-way",
        "base_config": "output/boundary_experiments_20260807/architecture_sweep/arch_oneway/config.yml",
        "checkpoint": "./output/boundary_experiments_20260807/architecture_sweep/arch_oneway/latest",
        "table9": True,
        "table10": False,
    },
    {
        "id": "dbm_d64",
        "name": "Full DBM, d=64",
        "base_config": "output/boundary_experiments_20260807/architecture_sweep/arch_dim64/config.yml",
        "checkpoint": "./output/boundary_experiments_20260807/architecture_sweep/arch_dim64/latest",
        "table9": True,
        "table10": False,
    },
    {
        "id": "dbm_ds8",
        "name": "Full DBM, d_s=8",
        "base_config": "output/boundary_experiments_20260807/architecture_sweep/arch_state8/config.yml",
        "checkpoint": "./output/boundary_experiments_20260807/architecture_sweep/arch_state8/latest",
        "table9": True,
        "table10": False,
    },
    {
        "id": "dbm_d192",
        "name": "Full DBM, d=192",
        "base_config": "output/boundary_experiments_20260807/architecture_sweep/arch_dim192/config.yml",
        "checkpoint": "./output/boundary_experiments_20260807/architecture_sweep/arch_dim192/latest",
        "table9": True,
        "table10": False,
    },
    {
        "id": "dbm_e1",
        "name": "Full DBM, e=1",
        "base_config": "output/boundary_experiments_20260807/architecture_sweep/arch_expand1/config.yml",
        "checkpoint": "./output/boundary_experiments_20260807/architecture_sweep/arch_expand1/latest",
        "table9": True,
        "table10": False,
    },
    {
        "id": "zero_geometry",
        "name": "Full Mamba, zero geometry",
        "base_config": "output/E11F_full_mamba_lambda02_geomzero_30000_from_baseline_20260709/config.yml",
        "checkpoint": "./output/E11F_full_mamba_lambda02_geomzero_30000_from_baseline_20260709/best_accuracy",
        "table9": True,
        "table10": False,
    },
    {
        "id": "true_horizontal",
        "name": "True Horizontal-only",
        "base_config": "output/boundary_experiments_20260807/architecture_sweep/arch_horizontal_true/config.yml",
        "checkpoint": "./output/boundary_experiments_20260807/architecture_sweep/arch_horizontal_true/latest",
        "table9": True,
        "table10": True,
    },
    {
        "id": "true_vertical",
        "name": "True Vertical-only",
        "base_config": "output/boundary_experiments_20260807/architecture_sweep/arch_vertical_true/config.yml",
        "checkpoint": "./output/boundary_experiments_20260807/architecture_sweep/arch_vertical_true/latest",
        "table9": True,
        "table10": True,
    },
    {
        "id": "dbm_slanet",
        "name": "DBM-SLANet",
        "base_config": "configs/table/DBM_SLANet_phase5e_eval_cta500_codex.yml",
        "checkpoint": "./output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy",
        "table9": False,
        "table10": True,
    },
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_eval_config(spec: dict) -> Path:
    cfg = yaml.safe_load((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    run_dir = OUT / spec["id"]
    cfg["Global"]["save_model_dir"] = str(run_dir)
    cfg["Global"]["save_res_path"] = str(run_dir / "infer")
    cfg["Global"]["diagnostics_path"] = str(run_dir / "sample_teds.jsonl")
    cfg["Global"]["pretrained_model"] = None
    cfg["Global"]["checkpoints"] = spec["checkpoint"]
    cfg["Eval"]["dataset"]["data_dir"] = VAL_DIR
    cfg["Eval"]["dataset"]["label_file_list"] = [str(PUB5000.relative_to(ROOT))]
    cfg["Eval"]["loader"]["batch_size_per_card"] = 4
    cfg["Eval"]["loader"]["num_workers"] = 0
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{spec['id']}.yml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def parse_last(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def run_eval(spec: dict, config_path: Path) -> dict:
    run_dir = OUT / spec["id"]
    sample_path = run_dir / "sample_teds.jsonl"
    log_path = LOG_DIR / f"{spec['id']}.eval.log"
    if sample_path.exists() and sum(1 for _ in sample_path.open(encoding="utf-8")) == 5000:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        return {
            "status": "completed_cached",
            "teds": parse_last(r"teds:([0-9.]+)", text),
            "structure_acc": parse_last(r"structure_acc:([0-9.]+)", text),
            "samples": 5000,
            "fps": parse_last(r"fps:([0-9.]+)", text),
            "log": str(log_path.relative_to(ROOT)),
            "sample_path": str(sample_path.relative_to(ROOT)),
        }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(PYTHON), "eval_table_teds_diagnostics_codex.py", "-c", str(config_path)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    wall = time.time() - start
    text = log_path.read_text(encoding="utf-8", errors="replace")
    status = "completed" if proc.returncode == 0 and "structure TEDS diagnostics eval" in text else "failed"
    return {
        "status": status,
        "return_code": proc.returncode,
        "wall_time_s": round(wall, 3),
        "teds": parse_last(r"teds:([0-9.]+)", text),
        "structure_acc": parse_last(r"structure_acc:([0-9.]+)", text),
        "samples": parse_last(r"samples:([0-9.]+)", text),
        "fps": parse_last(r"fps:([0-9.]+)", text),
        "log": str(log_path.relative_to(ROOT)),
        "sample_path": str(sample_path.relative_to(ROOT)),
    }


def profile_model(spec: dict, config_path: Path) -> dict:
    from ppocr.modeling.architectures import build_model

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_model(cfg["Architecture"])
    params = sum(int(x.numel()) for x in model.parameters())
    flops = paddle.flops(model, input_size=[1, 3, 488, 488], print_detail=False)
    head = cfg["Architecture"].get("Head", {})
    return {
        "model": spec["name"],
        "id": spec["id"],
        "head": head.get("name", ""),
        "d": head.get("cell_token_dim", ""),
        "d_s": head.get("mamba_d_state", ""),
        "e": head.get("mamba_expand", ""),
        "axis": head.get("axis_mode", "dual" if "Mamba" in head.get("name", "") else "none"),
        "direction": "uni" if head.get("bidirectional") is False else "bi" if "Mamba" in head.get("name", "") else "none",
        "presence_weight": cfg.get("Loss", {}).get("cell_presence_weight", ""),
        "params": params,
        "params_m": round(params / 1_000_000, 6),
        "flops": flops,
        "flops_g": round(flops / 1_000_000_000, 6),
    }


def clean_tokens(tokens: list[str]) -> list[str]:
    skip = {"<html>", "</html>", "<body>", "</body>", "<thead>", "</thead>", "<tbody>", "</tbody>"}
    return [str(x) for x in tokens if str(x) not in skip]


def parse_span(token: str, name: str) -> int:
    match = re.search(name + r'="([0-9]+)"', token)
    return int(match.group(1)) if match else 1


def parse_cells(tokens: list[str]) -> set[tuple[int, int, int, int]]:
    cells: set[tuple[int, int, int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    row = -1
    col = 0
    in_row = False
    for raw in clean_tokens(tokens):
        token = str(raw)
        if token.startswith("<tr"):
            row += 1
            col = 0
            in_row = True
            continue
        if token.startswith("</tr"):
            in_row = False
            continue
        if not in_row:
            continue
        if token.startswith("<td") or token.startswith("<th"):
            while (row, col) in occupied:
                col += 1
            rowspan = max(1, parse_span(token, "rowspan"))
            colspan = max(1, parse_span(token, "colspan"))
            cells.add((row, col, rowspan, colspan))
            for rr in range(row, row + rowspan):
                for cc in range(col, col + colspan):
                    occupied.add((rr, cc))
            col += colspan
    return cells


def f1_from_sets(pred: set, gt: set) -> tuple[int, int, int, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    denom = 2 * tp + fp + fn
    return tp, fp, fn, (2 * tp / denom if denom else 1.0)


def summarize_rows(rows: list[dict]) -> dict:
    total_teds = 0.0
    exact = 0
    top_tp = top_fp = top_fn = 0
    span_tp = span_fp = span_fn = 0
    top_scores = []
    gt_span_samples = 0
    pred_span_samples = 0
    parse_bad = 0
    for row in rows:
        total_teds += float(row["teds"])
        exact += int(bool(row["exact"]))
        try:
            pred_cells = parse_cells(row["pred_tokens"])
            gt_cells = parse_cells(row["gt_tokens"])
        except Exception:
            parse_bad += 1
            pred_cells = set()
            gt_cells = set()
        tp, fp, fn, score = f1_from_sets(pred_cells, gt_cells)
        top_tp += tp
        top_fp += fp
        top_fn += fn
        top_scores.append(score)
        pred_spans = {c for c in pred_cells if c[2] > 1 or c[3] > 1}
        gt_spans = {c for c in gt_cells if c[2] > 1 or c[3] > 1}
        gt_span_samples += int(bool(gt_spans))
        pred_span_samples += int(bool(pred_spans))
        tp, fp, fn, _ = f1_from_sets(pred_spans, gt_spans)
        span_tp += tp
        span_fp += fp
        span_fn += fn
    n = max(len(rows), 1)
    top_den = 2 * top_tp + top_fp + top_fn
    span_den = 2 * span_tp + span_fp + span_fn
    return {
        "n": len(rows),
        "teds": total_teds / n,
        "structure_acc": exact / n,
        "topology_f1": 2 * top_tp / top_den if top_den else 1.0,
        "topology_mean": sum(top_scores) / n,
        "span_f1": 2 * span_tp / span_den if span_den else 1.0,
        "gt_span_samples": gt_span_samples,
        "pred_span_samples": pred_span_samples,
        "parse_bad": parse_bad,
    }


def load_sample_rows(spec: dict) -> list[dict]:
    sample_path = OUT / spec["id"] / "sample_teds.jsonl"
    return read_jsonl(sample_path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cta_names = {row["filename"] for row in read_jsonl(CTA500)}
    run_rows = []
    profile_rows = []
    for index, spec in enumerate(MODELS, start=1):
        print(f"[{index}/{len(MODELS)}] preparing {spec['id']}", flush=True)
        cfg_path = build_eval_config(spec)
        profile = profile_model(spec, cfg_path)
        result = run_eval(spec, cfg_path)
        profile["fps"] = result.get("fps")
        profile["latency_ms"] = round(1000 / result["fps"], 3) if result.get("fps") else None
        profile_rows.append(profile)
        run_rows.append({"model": spec["name"], "id": spec["id"], **result})
        write_csv(OUT / "run_status.csv", run_rows, sorted({k for row in run_rows for k in row}))
        write_csv(OUT / "table2_profile.csv", profile_rows, sorted({k for row in profile_rows for k in row}))
        if result["status"] == "failed":
            raise SystemExit(f"failed: {spec['id']}")
    pub_rows = []
    cta_rows = []
    table9_rows = []
    table10_rows = []
    for spec, profile in zip(MODELS, profile_rows):
        samples = load_sample_rows(spec)
        pub = summarize_rows(samples)
        cta = summarize_rows([row for row in samples if row.get("filename") in cta_names])
        fps = profile.get("fps")
        common = {
            "model": spec["name"],
            "id": spec["id"],
            "fps": fps,
            "latency_ms": round(1000 / fps, 3) if fps else None,
            "params_m": profile["params_m"],
            "flops_g": profile["flops_g"],
        }
        pub_rows.append({**common, **pub})
        cta_rows.append({**common, **cta})
        if spec["table9"]:
            table9_rows.append({**common, **cta})
        if spec["table10"]:
            table10_rows.append({**common, **cta})
    fields = sorted({k for row in pub_rows for k in row})
    write_csv(OUT / "table3_pubtabnet5000.csv", pub_rows, fields)
    write_csv(OUT / "table4_pubtabnet5000_comprehensive.csv", pub_rows, fields)
    write_csv(OUT / "table5_cta500_from_pub5000.csv", cta_rows, sorted({k for row in cta_rows for k in row}))
    write_csv(OUT / "table9_phase11_cta500.csv", table9_rows, sorted({k for row in table9_rows for k in row}))
    write_csv(OUT / "table10_true_axis_rerun.csv", table10_rows, sorted({k for row in table10_rows for k in row}))
    (OUT / "COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")
    print(f"completed {OUT}", flush=True)


if __name__ == "__main__":
    main()
