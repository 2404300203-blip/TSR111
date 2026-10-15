import json
import os

import yaml


ROOT = "/root/autodl-tmp/PaddleOCR"
OUT = os.path.join(ROOT, "output/deployment_export_20260812/financial")
BASELINES = {
    "pilot_heldout75": "configs/table/SLANet_fintabnet_c_pilot_heldout_after_ft_eval_codex.yml",
    "train2000_cached500": "configs/table/SLANet_fintabnet_c_doduytoan_train2000_cached_test500_eval_codex.yml",
    "train2000_company100": "configs/table/SLANet_fintabnet_c_doduytoan_train2000_company_heldout100_eval_codex.yml",
    "company_seed20260806": "output/eval_company_disjoint_20260807/configs/company_seed20260806_slanet.yml",
    "company_seed20260807": "output/eval_company_disjoint_20260807/configs/company_seed20260807_slanet.yml",
    "company_seed20260808": "output/eval_company_disjoint_20260807/configs/company_seed20260808_slanet.yml",
}


def write_variant(source, output, batch_size):
    with open(source, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["Eval"]["loader"]["batch_size_per_card"] = batch_size
    config["Eval"]["loader"]["drop_last"] = False
    config["Eval"]["loader"]["shuffle"] = False
    config["Eval"]["loader"]["num_workers"] = 0
    with open(output, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def main():
    configs = []
    for model_id, relative in BASELINES.items():
        baseline_source = os.path.join(ROOT, relative)
        deploy_source = os.path.join(OUT, model_id + "_deploy.yml")
        for family, source in (("slanet", baseline_source), ("deploy", deploy_source)):
            for batch_size in (1, 4):
                path = os.path.join(
                    OUT, "%s_%s_bs%d.yml" % (model_id, family, batch_size)
                )
                write_variant(source, path, batch_size)
                configs.append(
                    {
                        "id": model_id,
                        "family": family,
                        "batch_size": batch_size,
                        "config": path,
                    }
                )
    with open(os.path.join(OUT, "timing_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(configs, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(len(configs))


if __name__ == "__main__":
    main()
