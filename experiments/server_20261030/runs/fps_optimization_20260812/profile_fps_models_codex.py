import csv
import gc
import json
import os
import sys

import paddle
import yaml


ROOT = "/root/autodl-tmp/PaddleOCR"
OUT = os.path.join(ROOT, "output/fps_optimization_20260812")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from ppocr.modeling.architectures import build_model


MAIN = os.path.join(ROOT, "output/main_tables_rebuild_20260810/configs")
CFG = os.path.join(OUT, "configs")
SPECS = [
    ("slanet", os.path.join(CFG, "slanet.yml"), False),
    ("dbm_full", os.path.join(CFG, "dbm_full.yml"), False),
    ("dbm_pruned", os.path.join(CFG, "dbm_pruned.yml"), True),
    ("dual_one_way", os.path.join(MAIN, "full_mamba_one_way.yml"), False),
    ("dual_one_way_pruned", os.path.join(MAIN, "full_mamba_one_way.yml"), True),
    ("true_horizontal", os.path.join(MAIN, "true_horizontal.yml"), False),
    ("true_horizontal_pruned", os.path.join(MAIN, "true_horizontal.yml"), True),
    ("true_vertical", os.path.join(MAIN, "true_vertical.yml"), False),
    ("true_vertical_pruned", os.path.join(MAIN, "true_vertical.yml"), True),
    ("d64", os.path.join(MAIN, "dbm_d64.yml"), False),
    ("d64_pruned", os.path.join(MAIN, "dbm_d64.yml"), True),
    ("ds8", os.path.join(MAIN, "dbm_ds8.yml"), False),
    ("ds8_pruned", os.path.join(MAIN, "dbm_ds8.yml"), True),
    ("d192", os.path.join(MAIN, "dbm_d192.yml"), False),
    ("d192_pruned", os.path.join(MAIN, "dbm_d192.yml"), True),
    ("expansion1", os.path.join(MAIN, "dbm_e1.yml"), False),
    ("expansion1_pruned", os.path.join(MAIN, "dbm_e1.yml"), True),
    ("three_scan_hbi_vuni", os.path.join(CFG, "three_scan_hbi_vuni_eval_full.yml"), False),
    ("three_scan_hbi_vuni_pruned", os.path.join(CFG, "three_scan_hbi_vuni_eval_pruned.yml"), True),
    ("three_scan_huni_vbi", os.path.join(CFG, "three_scan_huni_vbi_eval_full.yml"), False),
    ("three_scan_huni_vbi_pruned", os.path.join(CFG, "three_scan_huni_vbi_eval_pruned.yml"), True),
    ("shared_4dir", os.path.join(CFG, "shared_4dir_eval_full.yml"), False),
    ("shared_4dir_pruned", os.path.join(CFG, "shared_4dir_eval_pruned.yml"), True),
]

AUXILIARY_NAMES = (
    "cell_token_encoder",
    "cell_presence_head",
    "horizontal_mamba",
    "horizontal_reverse_mamba",
    "vertical_mamba",
    "vertical_reverse_mamba",
    "shared_mamba",
    "direction_embedding",
    "fusion",
)


def is_auxiliary_parameter(name):
    return any(part in name for part in AUXILIARY_NAMES)


def main():
    paddle.set_device("gpu:0")
    rows = []
    for model_id, config_path, pruned in SPECS:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        architecture = config["Architecture"]
        architecture["Backbone"]["pretrained"] = False
        if pruned and architecture["Head"]["name"] == "CellFullMambaHead":
            architecture["Head"]["prune_aux_at_inference"] = True

        model = build_model(architecture)
        model.eval()
        named_parameters = list(model.named_parameters())
        total_params = sum(int(parameter.numel()) for _, parameter in named_parameters)
        aux_params = sum(
            int(parameter.numel())
            for name, parameter in named_parameters
            if is_auxiliary_parameter(name)
        )
        active_params = total_params - aux_params if pruned else total_params
        flops = int(
            paddle.flops(
                model, input_size=[1, 3, 488, 488], print_detail=False
            )
        )
        head = architecture["Head"]
        rows.append(
            {
                "id": model_id,
                "head": head["name"],
                "axis_mode": head.get("axis_mode", "none"),
                "bidirectional": head.get("bidirectional", ""),
                "horizontal_bidirectional": head.get("horizontal_bidirectional", ""),
                "vertical_bidirectional": head.get("vertical_bidirectional", ""),
                "share_directional_mamba": head.get("share_directional_mamba", False),
                "pruned": pruned,
                "total_params": total_params,
                "auxiliary_params": aux_params,
                "active_params": active_params,
                "flops": flops,
                "total_params_m": total_params / 1_000_000.0,
                "active_params_m": active_params / 1_000_000.0,
                "flops_g": flops / 1_000_000_000.0,
                "config": config_path,
            }
        )
        del model
        paddle.device.cuda.empty_cache()
        gc.collect()

    json_path = os.path.join(OUT, "profiles/model_profiles.json")
    csv_path = os.path.join(OUT, "profiles/model_profiles.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
