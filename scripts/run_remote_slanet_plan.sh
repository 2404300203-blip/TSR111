#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/PaddleOCR}
PY=${PY:-/root/miniconda3/envs/slanet/bin/python}
CTA_DIR="$ROOT/train_data/table/pubtabnet_cta_benchmark_v1"

cd "$ROOT"

mkdir -p "$CTA_DIR"

"$PY" scripts/create_cta_pilot.py \
  --input "$CTA_DIR/cta_benchmark_v1_500.jsonl" \
  --output "$CTA_DIR/cta_benchmark_pilot_100.jsonl" \
  --manifest "$CTA_DIR/pilot_100_manifest.jsonl" \
  --summary "$CTA_DIR/pilot_100_dataset_summary.json" \
  --quality-report "$CTA_DIR/pilot_100_quality_report.md" \
  --limit 100

"$PY" scripts/validate_cta_dataset.py "$CTA_DIR/cta_benchmark_pilot_100.jsonl"

"$PY" scripts/prepare_transformer_only_config.py \
  --source configs/table/SLANet_pubtabnet_attn_transformer_codex.yml \
  --output configs/table/SLANet_pubtabnet_transformer_only_codex.yml

echo "Prepared CTA pilot and transformer-only config."
echo "Run training in screen/nohup:"
echo "$PY tools/train.py -c configs/table/SLANet_pubtabnet_attn_transformer_codex.yml"
echo "$PY tools/train.py -c configs/table/SLANet_pubtabnet_transformer_only_codex.yml"
