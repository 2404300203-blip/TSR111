#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/PaddleOCR
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/slanet/bin/python}
CFG=configs/table/SLANet_plus_E3_mamba_local_parallel_full_codex.yml
OUT=output/SLANet_plus_full/E3_mamba_local_parallel
mkdir -p "$OUT"
echo "[$(date '+%F %T')] start full E3: $CFG" | tee "$OUT/full_train_status.log"
"$PYTHON_BIN" tools/train.py -c "$CFG" 2>&1 | tee "$OUT/full_train_stdout.log"
"$PYTHON_BIN" scripts/summarize_slanet_plus_results.py || true
echo "[$(date '+%F %T')] full E3 finished" | tee -a "$OUT/full_train_status.log"
