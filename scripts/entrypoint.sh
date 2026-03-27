#!/bin/bash

set -e  # exit nếu có lỗi

OHLCV_DATASET="/notebooks/VN30F1M/VN30F1M_5m.csv"
FEATURE_DATASET="/notebooks/VN30F1M/VN30F1M_5m_features.csv"
OHLCV_URL="https://raw.githubusercontent.com/tempusoneps/vn-stock-data/refs/heads/main/VN30F1M/data_ohlcv/VN30F1M_5m.csv"

echo "=== START DATA PREP ==="

# 1. Check OHLCV_DATASET
if [ ! -f "$OHLCV_DATASET" ]; then
  echo "[INFO] OHLCV_DATASET chưa tồn tại. Đang download..."

  # đảm bảo thư mục tồn tại
  mkdir -p "$(dirname "$OHLCV_DATASET")"

  curl -o "$OHLCV_DATASET" "$OHLCV_URL"

  echo "[PASS] Đã download OHLCV_DATASET"
else
  echo "[PASS] OHLCV_DATASET đã tồn tại"
fi

# 2. Check FEATURE_DATASET
if [ ! -f "$FEATURE_DATASET" ]; then
  echo "[INFO] FEATURE_DATASET chưa tồn tại. Đang tạo..."

  autofcholv extract "$OHLCV_DATASET" --output "$FEATURE_DATASET"

  echo "[DONE] Đã tạo FEATURE_DATASET"
else
  echo "[SKIP] FEATURE_DATASET đã tồn tại"
fi


echo "=== DONE ==="

echo "=== START JUPYTER LAB ==="

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=''