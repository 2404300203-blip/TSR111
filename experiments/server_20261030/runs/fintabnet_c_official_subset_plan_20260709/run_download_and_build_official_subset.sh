#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/PaddleOCR
OUT=output/fintabnet_c_official_subset_plan_20260709
PY=/root/miniconda3/bin/python
export PYTHONPATH=.
mkdir -p "$OUT"
{
  echo "[$(date '+%F %T')] START download missing PDFs"
  $PY scripts/download_fintabnet_c_missing_pdfs.py \
    --lists \
      "$OUT/train_target_2000_missing_pdfs.txt" \
      "$OUT/val_target_300_missing_pdfs.txt" \
      "$OUT/test_target_500_missing_pdfs.txt" \
    --pdf-root train_data/table/fintabnet_processed/pdf_cache \
    --kaggle-bin /root/miniconda3/bin/kaggle \
    --retries 3 \
    --sleep 0.5
  echo "[$(date '+%F %T')] END download missing PDFs"

  echo "[$(date '+%F %T')] START build train2000"
  $PY scripts/build_fintabnet_c_paddleocr.py \
    --annotations-tar train_data/table/fintabnet_raw/fintabnet_c/FinTabNet.c-PDF_Annotations.tar.gz \
    --output-dir train_data/table/fintabnet_processed/fintabnet_c_official_train2000_val300_test500/train2000 \
    --pdf-root train_data/table/fintabnet_processed/pdf_cache \
    --split-name train --limit 2000 --sampling round_robin_company \
    --max-structure-len 500 --require-existing-pdf --pdftoppm-bin /usr/bin/pdftoppm

  echo "[$(date '+%F %T')] START build val300"
  $PY scripts/build_fintabnet_c_paddleocr.py \
    --annotations-tar train_data/table/fintabnet_raw/fintabnet_c/FinTabNet.c-PDF_Annotations.tar.gz \
    --output-dir train_data/table/fintabnet_processed/fintabnet_c_official_train2000_val300_test500/val300 \
    --pdf-root train_data/table/fintabnet_processed/pdf_cache \
    --split-name val --limit 300 --sampling round_robin_company \
    --max-structure-len 500 --require-existing-pdf --pdftoppm-bin /usr/bin/pdftoppm

  echo "[$(date '+%F %T')] START build test500"
  $PY scripts/build_fintabnet_c_paddleocr.py \
    --annotations-tar train_data/table/fintabnet_raw/fintabnet_c/FinTabNet.c-PDF_Annotations.tar.gz \
    --output-dir train_data/table/fintabnet_processed/fintabnet_c_official_train2000_val300_test500/test500 \
    --pdf-root train_data/table/fintabnet_processed/pdf_cache \
    --split-name test --limit 500 --sampling round_robin_company \
    --max-structure-len 500 --require-existing-pdf --pdftoppm-bin /usr/bin/pdftoppm

  echo "[$(date '+%F %T')] DONE build official subset"
} > "$OUT/download_and_build.log" 2>&1
