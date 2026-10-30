#!/bin/bash
set -e
cd /root/autodl-tmp/PaddleOCR
mkdir -p output/fair_controls_20260806/logs
for name in mamba_dual_bidir mamba_dual_unidir mamba_horizontal_bidir mamba_vertical_bidir slanet_loc10; do
  echo START $name | tee -a output/fair_controls_20260806/logs/all_train_restart.log
  rm -rf output/fair_controls_20260806/$name
  /root/miniconda3/envs/slanet/bin/python tools/train.py -c output/fair_controls_20260806/configs/${name}.yml > output/fair_controls_20260806/logs/${name}.restart.train.log 2>&1
  echo DONE $name | tee -a output/fair_controls_20260806/logs/all_train_restart.log
done
