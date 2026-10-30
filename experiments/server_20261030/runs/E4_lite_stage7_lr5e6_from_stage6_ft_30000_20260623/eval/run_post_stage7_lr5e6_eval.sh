#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/PaddleOCR
PY=/root/miniconda3/bin/python
CFG=configs/table/DBM_SLANet_full_mamba_lite_stage7_30000_lr5e6_from_stage6_codex.yml
OUT=output/E4_lite_stage7_lr5e6_from_stage6_ft_30000_20260623
TRAIN_PID=$(cat "$OUT/train.pid")
WATCH_LOG="$OUT/eval/post_stage7_lr5e6_eval.watch.log"
echo "[watcher:post_stage7_lr5e6_eval] started $(date) train_pid=$TRAIN_PID" >> "$WATCH_LOG"
while ps -p "$TRAIN_PID" >/dev/null 2>&1; do
  STEP=$(grep -o "global_step: [0-9]*" "$OUT/train.log" 2>/dev/null | tail -1 | awk "{print \$2}" || true)
  echo "[watcher:post_stage7_lr5e6_eval] $(date) training step=${STEP:-NA}" >> "$WATCH_LOG"
  sleep 300
done
echo "[watcher:post_stage7_lr5e6_eval] training stopped $(date), launching external TEDS" >> "$WATCH_LOG"
"$PY" eval_table_teds_codex.py \
  -c "$CFG" \
  -o Global.pretrained_model="$OUT/best_accuracy.pdparams" Global.checkpoints= \
  > "$OUT/eval/structure_teds_eval.log" 2>&1
echo "[watcher:post_stage7_lr5e6_eval] eval finished $(date)" >> "$WATCH_LOG"
