#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/PaddleOCR
OUT=output/fintabnet_c_official_subset_plan_20260709
PY=/root/miniconda3/bin/python
ZIP_DIR=train_data/table/fintabnet_raw/kaggle_full
ZIP_PATH=$ZIP_DIR/fintabnet.zip
export PYTHONPATH=.
mkdir -p "$OUT" "$ZIP_DIR"
{
  echo "[$(date '+%F %T')] START full Kaggle zip download"
  if [ ! -s "$ZIP_PATH" ]; then
    /root/miniconda3/bin/kaggle datasets download -d jiongjiong/fintabnet -p "$ZIP_DIR" --force
  else
    echo "zip already exists: $ZIP_PATH"
  fi
  echo "[$(date '+%F %T')] END full Kaggle zip download"
  ls -lh "$ZIP_PATH" || true

  echo "[$(date '+%F %T')] START extract needed PDFs"
  $PY scripts/extract_fintabnet_needed_pdfs_from_zip.py \
    --zip "$ZIP_PATH" \
    --lists \
      "$OUT/train_target_2000_missing_pdfs.txt" \
      "$OUT/val_target_300_missing_pdfs.txt" \
      "$OUT/test_target_500_missing_pdfs.txt" \
    --pdf-root train_data/table/fintabnet_processed/pdf_cache \
    --manifest "$OUT/extract_manifest.json"
  echo "[$(date '+%F %T')] END extract needed PDFs"

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

  echo "[$(date '+%F %T')] DONE fullzip extract and build"
} > "$OUT/fullzip_extract_and_build.log" 2>&1
