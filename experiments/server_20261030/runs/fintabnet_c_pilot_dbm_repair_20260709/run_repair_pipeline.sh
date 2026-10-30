#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/PaddleOCR
OUT=output/fintabnet_c_pilot_dbm_repair_20260709
PY=/root/miniconda3/bin/python
export PYTHONPATH=.
mkdir -p "$OUT"
run_one() {
  local ID="$1"
  local TRAIN_CFG="$2"
  local EVAL_CFG="$3"
  echo "[$(date '+%F %T')] START train $ID"
  $PY tools/train.py -c "$TRAIN_CFG" 2>&1 | tee "$OUT/${ID}_train.log"
  local train_status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END train $ID status=$train_status"
  echo "[$(date '+%F %T')] START heldout eval $ID"
  $PY scripts/eval_table_teds_codex.py -c "$EVAL_CFG" 2>&1 | tee "$OUT/${ID}_heldout_eval.log"
  local eval_status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END heldout eval $ID status=$eval_status"
}
{
  echo "[$(date '+%F %T')] DBM repair pipeline started"
  run_one "R1_dbm_phase5e_lr1e5_loc2_cell0" \
    configs/table/DBM_SLANet_fintabnet_c_pilot_repair_R1_dbm_phase5e_lr1e5_loc2_cell0_codex.yml \
    configs/table/DBM_SLANet_fintabnet_c_pilot_repair_R1_dbm_phase5e_lr1e5_loc2_cell0_heldout_eval_codex.yml
  run_one "R2_dbm_from_baselineft_lr1e5_loc2_cell0" \
    configs/table/DBM_SLANet_fintabnet_c_pilot_repair_R2_dbm_from_baselineft_lr1e5_loc2_cell0_codex.yml \
    configs/table/DBM_SLANet_fintabnet_c_pilot_repair_R2_dbm_from_baselineft_lr1e5_loc2_cell0_heldout_eval_codex.yml
  echo "[$(date '+%F %T')] SUMMARY"
  grep -RniE "teds:|structure_acc:|samples:|fps:" "$OUT"/*heldout_eval.log || true
} > "$OUT/repair_pipeline_driver.log" 2>&1
