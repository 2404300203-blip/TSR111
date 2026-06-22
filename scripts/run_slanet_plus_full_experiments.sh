#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/PaddleOCR

CONFIGS=(
  configs/table/SLANet_plus_E0_baseline_codex.yml
  configs/table/SLANet_plus_E1_local_only_codex.yml
  configs/table/SLANet_plus_E2_mamba_only_codex.yml
  configs/table/SLANet_plus_E3_mamba_local_parallel_codex.yml
  configs/table/SLANet_plus_E4_mamba_local_no_gate_codex.yml
  configs/table/SLANet_plus_E5_fsam_stable_codex.yml
)

PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/slanet/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=$(command -v python)
fi

for cfg in "${CONFIGS[@]}"; do
  out_dir=$(grep -m1 'save_model_dir:' "$cfg" | awk '{print $2}')
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] start $cfg -> $out_dir"
  "$PYTHON_BIN" tools/train.py -c "$cfg" 2>&1 | tee "$out_dir/train_stdout.log"
  if grep -q 'nanxxx' "$out_dir/train.log" "$out_dir/train_stdout.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] WARNING: nan detected in $cfg" >&2
  fi
  echo "[$(date '+%F %T')] done $cfg"
done

"$PYTHON_BIN" scripts/summarize_slanet_plus_results.py
