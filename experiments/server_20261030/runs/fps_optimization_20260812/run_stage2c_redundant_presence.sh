#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/PaddleOCR
OUT="$ROOT/output/fps_optimization_20260812"
PY=/root/miniconda3/envs/slanet/bin/python

cd "$ROOT"
while [[ ! -f "$OUT/stage2b_completed.txt" ]]; do
  sleep 15
done

date -Is > "$OUT/stage2c_started.txt"
"$PY" eval_table_teds_diagnostics_codex.py \
  -c "$OUT/configs/dbm_full_no_redundant_presence.yml" \
  > "$OUT/logs/dbm_full_no_redundant_presence.eval_pub5000.log" 2>&1
date -Is > "$OUT/stage2c_completed.txt"
