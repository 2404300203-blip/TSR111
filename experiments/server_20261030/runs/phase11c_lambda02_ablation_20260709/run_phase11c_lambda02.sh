#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/PaddleOCR
export PYTHONPATH=.
LOGDIR=output/phase11c_lambda02_ablation_20260709
PY=/root/miniconda3/bin/python
{
  echo "[$(date '+%F %T')] START phase11c full mamba lambda_pres=0.2 training"
  "$PY" tools/train.py -c configs/table/DBM_SLANet_phase11c_full_mamba_lambda02_30000_from_baseline_codex.yml 2>&1 | tee "$LOGDIR/phase11c_lambda02_train.log"
  echo "[$(date '+%F %T')] START phase11c CTA500 TEDS eval"
  "$PY" tools/eval.py -c configs/table/DBM_SLANet_phase11c_full_mamba_lambda02_eval_cta500_codex.yml 2>&1 | tee "$LOGDIR/phase11c_lambda02_cta500_eval.log"
  echo "[$(date '+%F %T')] DONE phase11c lambda02 ablation"
} > "$LOGDIR/phase11c_lambda02_driver.log" 2>&1
