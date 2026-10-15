import json
import os
import sys

import paddle
import yaml


ROOT = "/root/autodl-tmp/PaddleOCR"
OUT = os.path.join(ROOT, "output/deployment_export_20260812/financial")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ppocr.modeling.architectures import build_model


SPECS = [
    (
        "pilot_heldout75",
        "configs/table/DBM_SLANet_fintabnet_c_pilot_repair_R2_dbm_from_baselineft_lr1e5_loc2_cell0_heldout_eval_codex.yml",
    ),
    (
        "train2000_cached500",
        "configs/table/DBM_SLANet_fintabnet_c_doduytoan_train2000_r2_cached_test500_eval_codex.yml",
    ),
    (
        "train2000_company100",
        "configs/table/DBM_SLANet_fintabnet_c_doduytoan_train2000_r2_company_heldout100_eval_codex.yml",
    ),
    (
        "company_seed20260806",
        "output/eval_company_disjoint_20260807/configs/company_seed20260806_dbm.yml",
    ),
    (
        "company_seed20260807",
        "output/eval_company_disjoint_20260807/configs/company_seed20260807_dbm.yml",
    ),
    (
        "company_seed20260808",
        "output/eval_company_disjoint_20260807/configs/company_seed20260808_dbm.yml",
    ),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    manifest = []
    for model_id, relative_config in SPECS:
        source_config_path = os.path.join(ROOT, relative_config)
        with open(source_config_path, "r", encoding="utf-8") as stream:
            source_config = yaml.safe_load(stream)
        source_prefix = source_config["Global"]["checkpoints"]
        source_prefix = os.path.abspath(os.path.join(ROOT, source_prefix))
        source_checkpoint = source_prefix + ".pdparams"

        full_config = source_config.copy()
        full_config["Global"] = source_config["Global"].copy()
        full_config["Global"]["checkpoints"] = source_prefix
        full_config["Global"]["diagnostics_path"] = os.path.join(
            OUT, "predictions", model_id + "_full.jsonl"
        )
        full_config_path = os.path.join(OUT, model_id + "_full.yml")
        with open(full_config_path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(full_config, stream, sort_keys=False)

        deploy_config = source_config.copy()
        deploy_config["Global"] = source_config["Global"].copy()
        deploy_config["Architecture"] = source_config["Architecture"].copy()
        deploy_config["Architecture"]["Backbone"] = source_config["Architecture"]["Backbone"].copy()
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
            "loc_weight": float(source_config.get("Loss", {}).get("loc_weight", 2.0)),
            "loc_loss": source_config.get("Loss", {}).get("loc_loss", "smooth_l1"),
        }
        deploy_prefix = os.path.join(OUT, model_id + "_deploy")
        deploy_config["Global"]["checkpoints"] = deploy_prefix
        deploy_config["Global"]["diagnostics_path"] = os.path.join(
            OUT, "predictions", model_id + "_deploy.jsonl"
        )

        model = build_model(deploy_config["Architecture"])
        target = model.state_dict()
        source = paddle.load(source_checkpoint)
        missing = sorted(set(target) - set(source))
        mismatch = sorted(
            key
            for key in target.keys() & source.keys()
            if list(target[key].shape) != list(source[key].shape)
        )
        if missing or mismatch:
            raise RuntimeError("%s missing=%r mismatch=%r" % (model_id, missing, mismatch))
        compact = {key: source[key] for key in target}
        paddle.save(compact, deploy_prefix + ".pdparams")
        deploy_config_path = os.path.join(OUT, model_id + "_deploy.yml")
        with open(deploy_config_path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(deploy_config, stream, sort_keys=False)
        manifest.append(
            {
                "id": model_id,
                "source_config": source_config_path,
                "source_checkpoint": source_checkpoint,
                "full_config": full_config_path,
                "deploy_config": deploy_config_path,
                "deploy_checkpoint": deploy_prefix + ".pdparams",
                "source_keys": len(source),
                "deploy_keys": len(compact),
                "removed_keys": len(source) - len(compact),
                "deploy_parameters": sum(int(p.numel()) for p in model.parameters()),
            }
        )
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
