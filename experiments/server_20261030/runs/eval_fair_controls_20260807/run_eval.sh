#!/bin/bash
set -u
cd /root/autodl-tmp/PaddleOCR
: > output/eval_fair_controls_20260807/logs/all_eval.log
for name in cell_token mamba_dual_bidir mamba_dual_unidir mamba_horizontal_bidir mamba_vertical_bidir slanet_loc10; do
  echo START $name | tee -a output/eval_fair_controls_20260807/logs/all_eval.log
  PYTHONPATH=/root/autodl-tmp/PaddleOCR /root/miniconda3/envs/slanet/bin/python scripts/eval_table_teds_codex.py -c output/eval_fair_controls_20260807/configs/${name}.yml > output/eval_fair_controls_20260807/logs/${name}.eval.log 2>&1
  rc=$?; echo RC $rc $name | tee -a output/eval_fair_controls_20260807/logs/all_eval.log
  tail -8 output/eval_fair_controls_20260807/logs/${name}.eval.log | tee -a output/eval_fair_controls_20260807/logs/all_eval.log
done
