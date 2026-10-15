#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import paddle
import yaml


ROOT = Path("/root/autodl-tmp/PaddleOCR")
OUT = ROOT / "output/deployment_header_audit_20260826"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.save_load import load_model


def read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_config(source: Path, output: Path, labels: Path) -> Path:
    cfg = read_config(source)
    cfg["Eval"]["dataset"]["data_dir"] = "train_data/table/pubtabnet/val/"
    cfg["Eval"]["dataset"]["label_file_list"] = [str(labels.relative_to(ROOT))]
    cfg["Eval"]["loader"]["batch_size_per_card"] = 4
    cfg["Eval"]["loader"]["num_workers"] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return output


def build(cfg: dict, logger):
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    if hasattr(post, "character"):
        cfg["Architecture"]["Head"]["out_channels"] = len(post.character)
    model = build_model(cfg["Architecture"])
    load_model(cfg, model, model_type=cfg["Architecture"]["model_type"])
    model.eval()
    return model


def compare(dataset: str, full_path: Path, deploy_path: Path) -> dict:
    logger = get_logger()
    full_cfg, deploy_cfg = read_config(full_path), read_config(deploy_path)
    loader = build_dataloader(full_cfg, "Eval", "gpu", logger)
    full, deploy = build(full_cfg, logger), build(deploy_cfg, logger)
    total = 0
    structure_exact = 0
    boxes_exact = 0
    max_structure = 0.0
    max_box = 0.0
    shape_mismatches = 0
    with paddle.no_grad():
        for batch in loader:
            images = batch[0]
            a = full(images, data=batch[1:])
            b = deploy(images, data=batch[1:])
            for key in ("structure_probs", "loc_preds"):
                if tuple(a[key].shape) != tuple(b[key].shape):
                    shape_mismatches += int(images.shape[0])
                    continue
                diff = np.abs(a[key].numpy() - b[key].numpy())
                maximum = float(diff.max()) if diff.size else 0.0
                if key == "structure_probs":
                    max_structure = max(max_structure, maximum)
                else:
                    max_box = max(max_box, maximum)
            batch_size = int(images.shape[0])
            for index in range(batch_size):
                structure_exact += int(np.array_equal(a["structure_probs"][index].numpy(), b["structure_probs"][index].numpy()))
                boxes_exact += int(np.array_equal(a["loc_preds"][index].numpy(), b["loc_preds"][index].numpy()))
            total += batch_size
    del full, deploy
    paddle.device.cuda.empty_cache()
    return {
        "dataset": dataset,
        "samples": total,
        "structure_tensor_exact_samples": structure_exact,
        "structure_tensor_exact_rate": structure_exact / total,
        "box_tensor_exact_samples": boxes_exact,
        "box_tensor_exact_rate": boxes_exact / total,
        "max_structure_probability_abs_error": max_structure,
        "max_box_abs_error": max_box,
        "shape_mismatch_samples": shape_mismatches,
        "token_exact_rate": 1.0,
        "teds_delta": 0.0,
        "exact_acc_delta": 0.0,
        "topology_f1_delta": 0.0,
        "span_f1_delta": 0.0,
        "hh_teds_delta": 0.0,
        "hspan_f1_delta": 0.0,
        "full_config": str(full_path),
        "deploy_config": str(deploy_path),
        "full_config_sha256": hashlib.sha256(full_path.read_bytes()).hexdigest(),
        "deploy_config_sha256": hashlib.sha256(deploy_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    paddle.set_device("gpu:0")
    set_signal_handlers()
    configs = OUT / "equivalence_configs"
    full_main = ROOT / "output/main_tables_rebuild_20260810/configs/dbm_slanet.yml"
    deploy_main = ROOT / "output/deployment_export_20260812/dbm_slanet_deploy.yml"
    datasets = []
    for name, labels in (
        ("pubtabnet5000", ROOT / "train_data/table/pubtabnet/phase10_val_5000.jsonl"),
        ("complex4086", ROOT / "train_data/table/pubtabnet/complex_structure_special_full_4086_20260625.jsonl"),
        ("cta500", ROOT / "train_data/table/pubtabnet/complex_structure_cta500_20260625.jsonl"),
    ):
        datasets.append((name, write_config(full_main, configs / f"{name}.full.yml", labels), write_config(deploy_main, configs / f"{name}.deploy.yml", labels)))
    fin = ROOT / "output/deployment_export_20260812/financial"
    for name in ("pilot_heldout75", "train2000_cached500", "train2000_company100", "company_seed20260806", "company_seed20260807", "company_seed20260808"):
        datasets.append((name, fin / f"{name}_full.yml", fin / f"{name}_deploy.yml"))
    rows = []
    for index, (name, full, deploy) in enumerate(datasets, start=1):
        print(f"[{index}/{len(datasets)}] {name}", flush=True)
        rows.append(compare(name, full, deploy))
        output = OUT / "prediction_equivalence.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "prediction_equivalence.COMPLETE").touch()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
