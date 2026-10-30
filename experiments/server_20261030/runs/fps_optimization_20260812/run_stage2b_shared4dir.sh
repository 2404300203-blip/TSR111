#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/PaddleOCR
OUT="$ROOT/output/fps_optimization_20260812"
PY=/root/miniconda3/envs/slanet/bin/python

cd "$ROOT"
while [[ ! -f "$OUT/stage2_completed.txt" ]]; do
  sleep 15
done

date -Is > "$OUT/stage2b_started.txt"
sha256sum "$OUT"/configs/shared_4dir_*.yml \
  ppocr/modeling/heads/cell_full_mamba_head.py \
  > "$OUT/profiles/stage2b_input_sha256.txt"

"$PY" tools/train.py -c "$OUT/configs/shared_4dir_train.yml" \
  > "$OUT/logs/shared_4dir.train_console.log" 2>&1
date -Is > "$OUT/logs/shared_4dir.train_completed.txt"

"$PY" eval_table_teds_diagnostics_codex.py \
  -c "$OUT/configs/shared_4dir_eval_full.yml" \
  > "$OUT/logs/shared_4dir.eval_full_pub5000.log" 2>&1

"$PY" eval_table_teds_diagnostics_codex.py \
  -c "$OUT/configs/shared_4dir_eval_pruned.yml" \
  > "$OUT/logs/shared_4dir.eval_pruned_pub5000.log" 2>&1

"$PY" eval_table_teds_diagnostics_codex.py \
  -c "$OUT/configs/shared_4dir_eval_cta500.yml" \
  > "$OUT/logs/shared_4dir.eval_cta500.log" 2>&1

date -Is > "$OUT/stage2b_completed.txt"
