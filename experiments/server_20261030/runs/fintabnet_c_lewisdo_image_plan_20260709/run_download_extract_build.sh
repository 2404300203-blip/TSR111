#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/PaddleOCR
OUT=output/fintabnet_c_lewisdo_image_plan_20260709
ZIP_DIR=train_data/table/fintabnet_raw/lewisdo_fintabnet
ZIP_PATH=$ZIP_DIR/fintabnet.zip
EXTRACT_DIR=train_data/table/fintabnet_raw/lewisdo_fintabnet/extracted
PY=/root/miniconda3/bin/python
export PYTHONPATH=.:scripts
mkdir -p "$OUT" "$ZIP_DIR" "$EXTRACT_DIR"
{
  echo "[$(date '+%F %T')] START lewisdo/fintabnet download"
  if [ ! -s "$ZIP_PATH" ]; then
    /root/miniconda3/bin/kaggle datasets download -d lewisdo/fintabnet -p "$ZIP_DIR" --force
  else
    echo "zip already exists: $ZIP_PATH"
  fi
  echo "[$(date '+%F %T')] END lewisdo/fintabnet download"
  ls -lh "$ZIP_PATH" || true

  echo "[$(date '+%F %T')] START unzip lewisdo images"
  if [ ! -d "$EXTRACT_DIR/FinTabNet.c-Structure/images" ]; then
    unzip -q "$ZIP_PATH" 'FinTabNet.c-Structure/images/*' -d "$EXTRACT_DIR"
  else
    echo "images already extracted"
  fi
  echo "[$(date '+%F %T')] END unzip lewisdo images"
  find "$EXTRACT_DIR/FinTabNet.c-Structure/images" -type f | wc -l

  echo "[$(date '+%F %T')] START build train2000 from images"
  $PY scripts/build_fintabnet_c_from_cropped_images.py \
    --annotations-tar train_data/table/fintabnet_raw/fintabnet_c/FinTabNet.c-PDF_Annotations.tar.gz \
    --image-root "$EXTRACT_DIR/FinTabNet.c-Structure/images" \
    --output-dir train_data/table/fintabnet_processed/fintabnet_c_lewisdo_train2000_val300_test500/train2000 \
    --split-name train --limit 2000 --sampling round_robin_company --max-structure-len 500

  echo "[$(date '+%F %T')] START build val300 from images"
  $PY scripts/build_fintabnet_c_from_cropped_images.py \
    --annotations-tar train_data/table/fintabnet_raw/fintabnet_c/FinTabNet.c-PDF_Annotations.tar.gz \
    --image-root "$EXTRACT_DIR/FinTabNet.c-Structure/images" \
    --output-dir train_data/table/fintabnet_processed/fintabnet_c_lewisdo_train2000_val300_test500/val300 \
    --split-name val --limit 300 --sampling round_robin_company --max-structure-len 500

  echo "[$(date '+%F %T')] START build test500 from images"
  $PY scripts/build_fintabnet_c_from_cropped_images.py \
    --annotations-tar train_data/table/fintabnet_raw/fintabnet_c/FinTabNet.c-PDF_Annotations.tar.gz \
    --image-root "$EXTRACT_DIR/FinTabNet.c-Structure/images" \
    --output-dir train_data/table/fintabnet_processed/fintabnet_c_lewisdo_train2000_val300_test500/test500 \
    --split-name test --limit 500 --sampling round_robin_company --max-structure-len 500

  echo "[$(date '+%F %T')] DONE lewisdo image build"
} > "$OUT/download_extract_build.log" 2>&1
