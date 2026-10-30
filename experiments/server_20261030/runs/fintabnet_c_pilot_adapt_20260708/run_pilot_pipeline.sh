#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/PaddleOCR
OUT=output/fintabnet_c_pilot_adapt_20260708
PY=/root/miniconda3/bin/python
export PYTHONPATH=.
mkdir -p "$OUT"
{
  echo "[$(date '+%F %T')] START baseline fine-tune"
  $PY tools/train.py -c configs/table/SLANet_fintabnet_c_pilot_ft_350_codex.yml 2>&1 | tee "$OUT/baseline_ft_train.log"
  status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END baseline fine-tune status=$status"

  echo "[$(date '+%F %T')] START baseline heldout after-ft eval"
  $PY scripts/eval_table_teds_codex.py -c configs/table/SLANet_fintabnet_c_pilot_heldout_after_ft_eval_codex.yml 2>&1 | tee "$OUT/baseline_ft_heldout_eval.log"
  status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END baseline heldout after-ft eval status=$status"

  echo "[$(date '+%F %T')] START DBM fine-tune"
  $PY tools/train.py -c configs/table/DBM_SLANet_phase5e_fintabnet_c_pilot_ft_350_codex.yml 2>&1 | tee "$OUT/dbm_phase5e_ft_train.log"
  status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END DBM fine-tune status=$status"

  echo "[$(date '+%F %T')] START DBM heldout after-ft eval"
  $PY scripts/eval_table_teds_codex.py -c configs/table/DBM_SLANet_phase5e_fintabnet_c_pilot_heldout_after_ft_eval_codex.yml 2>&1 | tee "$OUT/dbm_phase5e_ft_heldout_eval.log"
  status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END DBM heldout after-ft eval status=$status"

  echo "[$(date '+%F %T')] SUMMARY"
  grep -RniE "teds:|structure_acc:|samples:|fps:" "$OUT"/*heldout*eval.log || true
} > "$OUT/pipeline_driver.log" 2>&1
