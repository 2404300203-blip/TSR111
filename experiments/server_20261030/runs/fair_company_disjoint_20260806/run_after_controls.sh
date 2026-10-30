#!/bin/bash
set -e
cd /root/autodl-tmp/PaddleOCR
while kill -0 4117 2>/dev/null; do sleep 30; done
mkdir -p output/fair_company_disjoint_20260806/logs
for seed in 20260806 20260807 20260808; do
  for kind in slanet dbm; do
    name=company_seed${seed}_${kind}
    echo START $name | tee -a output/fair_company_disjoint_20260806/logs/company_all_train.log
    /root/miniconda3/envs/slanet/bin/python tools/train.py -c output/fair_company_disjoint_20260806/configs/${name}.yml > output/fair_company_disjoint_20260806/logs/${name}.train.log 2>&1
    echo DONE $name | tee -a output/fair_company_disjoint_20260806/logs/company_all_train.log
  done
done
