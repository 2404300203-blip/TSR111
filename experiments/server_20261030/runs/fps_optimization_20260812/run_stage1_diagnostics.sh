#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/PaddleOCR
OUT="$ROOT/output/fps_optimization_20260812"
PY=/root/miniconda3/envs/slanet/bin/python

cd "$ROOT"
date -Is > "$OUT/stage1_started.txt"
nvidia-smi -q > "$OUT/profiles/nvidia_smi_before.txt"
sha256sum "$OUT"/configs/*.yml ppocr/modeling/heads/cell_full_mamba_head.py \
  > "$OUT/profiles/input_sha256.txt"

for model in dbm_pruned dbm_full slanet; do
  "$PY" eval_table_teds_diagnostics_codex.py \
    -c "$OUT/configs/$model.yml" \
    > "$OUT/logs/$model.eval_pub5000.log" 2>&1
  date -Is > "$OUT/logs/$model.completed.txt"
done

nvidia-smi -q > "$OUT/profiles/nvidia_smi_after.txt"
date -Is > "$OUT/stage1_completed.txt"
