#!/bin/bash
set -u
cd /root/autodl-tmp/PaddleOCR
: > output/eval_company_disjoint_20260807/logs/all_eval.log
for name in company_seed20260806_slanet company_seed20260806_dbm company_seed20260807_slanet company_seed20260807_dbm company_seed20260808_slanet company_seed20260808_dbm; do
  echo START $name | tee -a output/eval_company_disjoint_20260807/logs/all_eval.log
  PYTHONPATH=/root/autodl-tmp/PaddleOCR /root/miniconda3/envs/slanet/bin/python scripts/eval_table_teds_codex.py -c output/eval_company_disjoint_20260807/configs/${name}.yml > output/eval_company_disjoint_20260807/logs/${name}.eval.log 2>&1
  rc=$?; echo RC $rc $name | tee -a output/eval_company_disjoint_20260807/logs/all_eval.log
  tail -8 output/eval_company_disjoint_20260807/logs/${name}.eval.log | tee -a output/eval_company_disjoint_20260807/logs/all_eval.log
done
