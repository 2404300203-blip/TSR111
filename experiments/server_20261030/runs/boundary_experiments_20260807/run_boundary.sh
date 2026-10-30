#!/bin/bash
set -e
cd /root/autodl-tmp/PaddleOCR
ROOT=output/boundary_experiments_20260807
: > $ROOT/logs/all_train.log
TRAIN_JOBS="fixed_company_trainseed20260807_slanet fixed_company_trainseed20260807_dbm fixed_company_trainseed20260808_slanet fixed_company_trainseed20260808_dbm loc2_slanet loc2_dbm loc5_slanet loc5_dbm loc20_slanet loc20_dbm"
for name in $TRAIN_JOBS; do
  echo START_TRAIN $name | tee -a $ROOT/logs/all_train.log
  /root/miniconda3/envs/slanet/bin/python tools/train.py -c $ROOT/configs/${name}.yml > $ROOT/logs/${name}.train.log 2>&1
  echo DONE_TRAIN $name | tee -a $ROOT/logs/all_train.log
done
: > $ROOT/logs/all_eval.log
EVAL_JOBS="fixed_company_trainseed20260806_slanet fixed_company_trainseed20260806_dbm fixed_company_trainseed20260807_slanet fixed_company_trainseed20260807_dbm fixed_company_trainseed20260808_slanet fixed_company_trainseed20260808_dbm loc2_slanet loc2_dbm loc5_slanet loc5_dbm loc10_slanet loc10_dbm loc20_slanet loc20_dbm"
for name in $EVAL_JOBS; do
  echo START_EVAL $name | tee -a $ROOT/logs/all_eval.log
  PYTHONPATH=/root/autodl-tmp/PaddleOCR /root/miniconda3/envs/slanet/bin/python scripts/eval_table_teds_codex.py -c $ROOT/eval_configs/${name}.yml > $ROOT/logs/${name}.eval.log 2>&1
  echo DONE_EVAL $name | tee -a $ROOT/logs/all_eval.log
  grep -E "teds:|structure_acc:|samples:|fps:" $ROOT/logs/${name}.eval.log | tail -4 | tee -a $ROOT/logs/all_eval.log
done
