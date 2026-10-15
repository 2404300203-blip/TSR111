import csv
import hashlib
import json
import os
import sys

import paddle
import yaml


ROOT = "/root/autodl-tmp/PaddleOCR"
SOURCE_CONFIG = os.path.join(ROOT, "output/fps_optimization_20260812/configs/dbm_full.yml")
SOURCE_CHECKPOINT = os.path.join(
    ROOT,
    "output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams",
)
OUT = os.path.join(ROOT, "output/deployment_export_20260812")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ppocr.modeling.architectures import build_model


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(SOURCE_CONFIG, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    deploy_config = config.copy()
    deploy_config["Architecture"] = config["Architecture"].copy()
    deploy_config["Architecture"]["Backbone"] = config["Architecture"]["Backbone"].copy()
    deploy_config["Architecture"]["Backbone"]["pretrained"] = False
    deploy_config["Architecture"]["Head"] = {
        "name": "SLAHead",
        "hidden_size": 256,
        "max_text_length": 500,
        "loc_reg_num": 4,
    }
    deploy_config["Loss"] = {
        "name": "SLALoss",
        "structure_weight": 1.0,
        "loc_weight": 10.0,
        "loc_loss": "smooth_l1",
    }

    checkpoint_prefix = os.path.join(OUT, "dbm_slanet_deploy")
    deploy_config["Global"]["checkpoints"] = checkpoint_prefix
    deploy_config["Global"]["save_model_dir"] = OUT
    deploy_config["Global"]["save_inference_dir"] = os.path.join(OUT, "inference")
    deploy_config["Global"]["diagnostics_path"] = os.path.join(
        OUT, "predictions", "pubtabnet5000_deploy.jsonl"
    )

    model = build_model(deploy_config["Architecture"])
    target_state = model.state_dict()
    source_state = paddle.load(SOURCE_CHECKPOINT)
    missing = sorted(set(target_state) - set(source_state))
    shape_mismatch = sorted(
        key
        for key in target_state.keys() & source_state.keys()
        if list(target_state[key].shape) != list(source_state[key].shape)
    )
    if missing or shape_mismatch:
        raise RuntimeError(
            "Cannot build exact compact checkpoint: missing=%r shape_mismatch=%r"
            % (missing, shape_mismatch)
        )

    compact_state = {key: source_state[key] for key in target_state}
    model.set_state_dict(compact_state)
    paddle.save(compact_state, checkpoint_prefix + ".pdparams")

    config_path = os.path.join(OUT, "dbm_slanet_deploy.yml")
    with open(config_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(deploy_config, stream, sort_keys=False)

    rows = []
    for name, parameter in model.named_parameters():
        rows.append(
            {
                "name": name,
                "shape": "x".join(str(value) for value in parameter.shape),
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype),
            }
        )
    manifest_path = os.path.join(OUT, "parameter_manifest.csv")
    with open(manifest_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    model.eval()
    flops = int(paddle.flops(model, input_size=[1, 3, 488, 488], print_detail=False))
    checkpoint_path = checkpoint_prefix + ".pdparams"
    summary = {
        "source_checkpoint": SOURCE_CHECKPOINT,
        "deploy_checkpoint": checkpoint_path,
        "config": config_path,
        "stored_parameters": sum(row["numel"] for row in rows),
        "trainable_parameters": sum(int(p.numel()) for p in model.parameters() if not p.stop_gradient),
        "paddle_flops_proxy": flops,
        "checkpoint_size_bytes": os.path.getsize(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "parameter_manifest_sha256": sha256(manifest_path),
        "source_parameter_keys": len(source_state),
        "deploy_parameter_keys": len(compact_state),
        "removed_parameter_keys": len(source_state) - len(compact_state),
        "missing_keys": missing,
        "shape_mismatch": shape_mismatch,
    }
    with open(os.path.join(OUT, "model_size.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
