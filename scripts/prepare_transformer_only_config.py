#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    text = Path(args.source).read_text(encoding="utf-8")
    replacements = {
        "model_name: SLANet_AttnTransformer": "model_name: SLANet_TransformerOnly",
        "save_model_dir: ./output/SLANet_pubtabnet_attn_transformer_codex": "save_model_dir: ./output/SLANet_pubtabnet_transformer_only_codex",
        "save_inference_dir: ./output/SLANet_pubtabnet_attn_transformer_codex/infer": "save_inference_dir: ./output/SLANet_pubtabnet_transformer_only_codex/infer",
        "save_res_path: 'output/SLANet_pubtabnet_attn_transformer_codex/infer'": "save_res_path: 'output/SLANet_pubtabnet_transformer_only_codex/infer'",
        "use_refine: True": "use_refine: False",
    }
    for old, new in replacements.items():
        if old not in text:
            raise SystemExit(f"Expected text not found in source config: {old}")
        text = text.replace(old, new, 1)

    Path(args.output).write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
