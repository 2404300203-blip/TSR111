#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/PaddleOCR
OUT=output/E4_lite_stage5_lr2e5_ft_30000_20260623
CFG=configs/table/DBM_SLANet_full_mamba_lite_stage5_30000_lr2e5_codex.yml
WATCH_LOG="$OUT/eval/post_stage5_lr2e5_eval.watch.log"
echo "[watcher:post_stage5_lr2e5_eval_unique] started $(date)" >> "$WATCH_LOG"
while ps -eo args | grep -F "tools/train.py -c ${CFG}" | grep -v grep >/dev/null; do
  step=$(grep -o "global_step: [0-9]*" "$OUT/train.log" | tail -1 | sed "s/[^0-9]//g")
  echo "[watcher:post_stage5_lr2e5_eval_unique] $(date) training step=${step:-NA}" >> "$WATCH_LOG"
  sleep 300
done
echo "[watcher:post_stage5_lr2e5_eval_unique] training stopped $(date), launching external TEDS" >> "$WATCH_LOG"
if [ -f "$OUT/best_accuracy.pdparams" ]; then
  /root/miniconda3/bin/python eval_table_teds_codex.py \
    -c "$CFG" \
    -o Global.pretrained_model="$OUT/best_accuracy.pdparams" Global.checkpoints= \
    > "$OUT/eval/structure_teds_eval.log" 2>&1
  echo "[watcher:post_stage5_lr2e5_eval_unique] eval finished $(date)" >> "$WATCH_LOG"
else
  echo "[watcher:post_stage5_lr2e5_eval_unique] missing best_accuracy.pdparams" >> "$WATCH_LOG"
fi
