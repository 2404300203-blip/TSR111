#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/PaddleOCR}
PY=${PY:-/root/miniconda3/envs/slanet/bin/python}
cd "$ROOT"

wait_for_checkpoint() {
  local name=$1
  local path=$2
  echo "[$(date)] waiting for $name checkpoint: $path"
  while [[ ! -f "${path}.pdparams" ]]; do
    sleep 120
  done
  echo "[$(date)] found $name checkpoint"
}

run_teds_eval() {
  local name=$1
  local config=$2
  local checkpoint=$3
  local log=$4
  echo "[$(date)] eval $name" | tee "$log"
  "$PY" scripts/eval_table_teds_codex.py \
    -c "$config" \
    -o Global.checkpoints="$checkpoint" 2>&1 | tee -a "$log"
}

wait_for_checkpoint \
  "attn transformer" \
  "./output/SLANet_pubtabnet_attn_transformer_codex/best_accuracy"

run_teds_eval \
  "attn transformer" \
  "configs/table/SLANet_pubtabnet_attn_transformer_codex.yml" \
  "./output/SLANet_pubtabnet_attn_transformer_codex/best_accuracy" \
  "eval_attn_transformer_teds.log"

wait_for_checkpoint \
  "transformer only" \
  "./output/SLANet_pubtabnet_transformer_only_codex/best_accuracy"

run_teds_eval \
  "transformer only" \
  "configs/table/SLANet_pubtabnet_transformer_only_codex.yml" \
  "./output/SLANet_pubtabnet_transformer_only_codex/best_accuracy" \
  "eval_transformer_only_teds.log"

echo "[$(date)] eval watcher done"
