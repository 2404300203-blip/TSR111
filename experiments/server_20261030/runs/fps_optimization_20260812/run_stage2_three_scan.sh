#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/PaddleOCR
OUT="$ROOT/output/fps_optimization_20260812"
PY=/root/miniconda3/envs/slanet/bin/python

cd "$ROOT"
while [[ ! -f "$OUT/stage1_completed.txt" ]]; do
  sleep 15
done

date -Is > "$OUT/stage2_started.txt"
sha256sum "$OUT"/configs/three_scan_*.yml \
  ppocr/modeling/heads/cell_full_mamba_head.py \
  > "$OUT/profiles/stage2_input_sha256.txt"

for model in three_scan_hbi_vuni three_scan_huni_vbi; do
  "$PY" tools/train.py -c "$OUT/configs/${model}_train.yml" \
    > "$OUT/logs/${model}.train_console.log" 2>&1
  date -Is > "$OUT/logs/${model}.train_completed.txt"

  "$PY" eval_table_teds_diagnostics_codex.py \
    -c "$OUT/configs/${model}_eval_full.yml" \
    > "$OUT/logs/${model}.eval_full_pub5000.log" 2>&1

  "$PY" eval_table_teds_diagnostics_codex.py \
    -c "$OUT/configs/${model}_eval_pruned.yml" \
    > "$OUT/logs/${model}.eval_pruned_pub5000.log" 2>&1

  "$PY" eval_table_teds_diagnostics_codex.py \
    -c "$OUT/configs/${model}_eval_cta500.yml" \
    > "$OUT/logs/${model}.eval_cta500.log" 2>&1
  date -Is > "$OUT/logs/${model}.eval_completed.txt"
done

date -Is > "$OUT/stage2_completed.txt"
