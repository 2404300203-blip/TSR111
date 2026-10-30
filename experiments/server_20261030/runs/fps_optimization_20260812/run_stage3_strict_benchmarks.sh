#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/PaddleOCR
OUT="$ROOT/output/fps_optimization_20260812"
PY=/root/miniconda3/envs/slanet/bin/python
BENCH="$OUT/benchmark_table_inference_codex.py"

cd "$ROOT"
while [[ ! -f "$OUT/stage2c_completed.txt" ]]; do
  sleep 15
done

date -Is > "$OUT/stage3_started.txt"

run_one() {
  local id="$1"
  local config="$2"
  local prune="$3"
  local batch_size="$4"
  local measured_batches="$5"
  local run="$6"
  local result="$OUT/timing/${id}.bs${batch_size}.run${run}.json"
  local log="$OUT/logs/${id}.bs${batch_size}.run${run}.timing.log"
  local command=(
    "$PY" "$BENCH" -c "$config" -o
    "Eval.loader.batch_size_per_card=$batch_size"
  )
  if [[ "$prune" == "true" ]]; then
    command+=("Architecture.Head.prune_aux_at_inference=true")
  fi
  FPS_BENCH_OUTPUT="$result" \
  FPS_BENCH_RUN="$run" \
  FPS_BENCH_WARMUP=20 \
  FPS_BENCH_BATCHES="$measured_batches" \
    "${command[@]}" > "$log" 2>&1
}

run_condition() {
  local id="$1"
  local config="$2"
  local prune="$3"
  local run
  for run in 1 2 3; do
    run_one "$id" "$config" "$prune" 4 200 "$run"
    run_one "$id" "$config" "$prune" 1 400 "$run"
  done
}

MAIN="$ROOT/output/main_tables_rebuild_20260810/configs"
CFG="$OUT/configs"

run_condition slanet "$CFG/slanet.yml" false
run_condition dbm_full "$CFG/dbm_full.yml" false
run_condition dbm_full_no_redundant_presence "$CFG/dbm_full_no_redundant_presence.yml" false
run_condition dbm_pruned "$CFG/dbm_pruned.yml" false
run_condition cell_token "$MAIN/cell_token.yml" false
run_condition dual_one_way "$MAIN/full_mamba_one_way.yml" false
run_condition dual_one_way_pruned "$MAIN/full_mamba_one_way.yml" true
run_condition true_horizontal "$MAIN/true_horizontal.yml" false
run_condition true_horizontal_pruned "$MAIN/true_horizontal.yml" true
run_condition true_vertical "$MAIN/true_vertical.yml" false
run_condition true_vertical_pruned "$MAIN/true_vertical.yml" true
run_condition d64 "$MAIN/dbm_d64.yml" false
run_condition d64_pruned "$MAIN/dbm_d64.yml" true
run_condition ds8 "$MAIN/dbm_ds8.yml" false
run_condition ds8_pruned "$MAIN/dbm_ds8.yml" true
run_condition d192 "$MAIN/dbm_d192.yml" false
run_condition d192_pruned "$MAIN/dbm_d192.yml" true
run_condition expansion1 "$MAIN/dbm_e1.yml" false
run_condition expansion1_pruned "$MAIN/dbm_e1.yml" true
run_condition three_scan_hbi_vuni "$CFG/three_scan_hbi_vuni_eval_full.yml" false
run_condition three_scan_hbi_vuni_pruned "$CFG/three_scan_hbi_vuni_eval_pruned.yml" false
run_condition three_scan_huni_vbi "$CFG/three_scan_huni_vbi_eval_full.yml" false
run_condition three_scan_huni_vbi_pruned "$CFG/three_scan_huni_vbi_eval_pruned.yml" false
run_condition shared_4dir "$CFG/shared_4dir_eval_full.yml" false
run_condition shared_4dir_pruned "$CFG/shared_4dir_eval_pruned.yml" false

date -Is > "$OUT/stage3_completed.txt"
