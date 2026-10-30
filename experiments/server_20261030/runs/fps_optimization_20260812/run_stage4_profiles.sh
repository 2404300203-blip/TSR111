#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/PaddleOCR
OUT="$ROOT/output/fps_optimization_20260812"
PY=/root/miniconda3/envs/slanet/bin/python

cd "$ROOT"
while [[ ! -f "$OUT/stage3_completed.txt" ]]; do
  sleep 15
done

date -Is > "$OUT/stage4_started.txt"
"$PY" "$OUT/profile_fps_models_codex.py" \
  > "$OUT/logs/model_profiles.log" 2>&1
date -Is > "$OUT/stage4_completed.txt"
