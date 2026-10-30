#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/PaddleOCR
OUT=output/fintabnet_c_doduytoan_train2000_r2_20260709
PY=/root/miniconda3/bin/python
export PYTHONPATH=.
mkdir -p "$OUT"
run_train() {
  local ID="$1"; local CFG="$2"
  echo "[$(date '+%F %T')] START train $ID"
  $PY tools/train.py -c "$CFG" 2>&1 | tee "$OUT/${ID}.log"
  local status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END train $ID status=$status"
}
run_eval() {
  local ID="$1"; local CFG="$2"
  echo "[$(date '+%F %T')] START eval $ID"
  $PY scripts/eval_table_teds_codex.py -c "$CFG" 2>&1 | tee "$OUT/${ID}.log"
  local status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END eval $ID status=$status"
}
{
  echo "[$(date '+%F %T')] doduytoan train2000 R2 pipeline started"
  run_train baseline_ft_train configs/table/SLANet_fintabnet_c_doduytoan_train2000_ft_codex.yml
  run_eval baseline_cached_test500_eval configs/table/SLANet_fintabnet_c_doduytoan_train2000_cached_test500_eval_codex.yml
  run_eval baseline_company_heldout100_eval configs/table/SLANet_fintabnet_c_doduytoan_train2000_company_heldout100_eval_codex.yml
  run_train dbm_r2_ft_train configs/table/DBM_SLANet_fintabnet_c_doduytoan_train2000_r2_ft_codex.yml
  run_eval dbm_r2_cached_test500_eval configs/table/DBM_SLANet_fintabnet_c_doduytoan_train2000_r2_cached_test500_eval_codex.yml
  run_eval dbm_r2_company_heldout100_eval configs/table/DBM_SLANet_fintabnet_c_doduytoan_train2000_r2_company_heldout100_eval_codex.yml
  echo "[$(date '+%F %T')] SUMMARY"
  grep -RniE "teds:|structure_acc:|samples:|fps:" "$OUT"/*eval.log || true
} > "$OUT/train2000_r2_pipeline_driver.log" 2>&1
