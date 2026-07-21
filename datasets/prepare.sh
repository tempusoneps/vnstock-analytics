#!/bin/bash

# Exit on error
set -e

# Determine the directory of this script to ensure relative path works from any working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OHLCV_DATASET="${SCRIPT_DIR}/VN30F1M_5m.csv"
FEATURE_DATASET="${SCRIPT_DIR}/VN30F1M_5m_features.csv"
LABEL_DATASET="${SCRIPT_DIR}/VN30F1M_5m_labels.csv"
OHLCV_URL="https://raw.githubusercontent.com/tempusoneps/vn-stock-data/refs/heads/main/VN30F1M/data_ohlcv/VN30F1M_5m.csv"

echo "=== START DATA PREP ==="

# 1. Check OHLCV_DATASET
if [ ! -f "$OHLCV_DATASET" ]; then
  echo "[INFO] OHLCV_DATASET chưa tồn tại. Đang download..."

  # Đảm bảo thư mục tồn tại
  mkdir -p "$(dirname "$OHLCV_DATASET")"

  curl -L -o "$OHLCV_DATASET" "$OHLCV_URL"

  echo "[PASS] Đã download OHLCV_DATASET"
else
  echo "[PASS] OHLCV_DATASET đã tồn tại"
fi

# 2. Check LABEL_DATASET
if [ ! -f "$LABEL_DATASET" ]; then
  echo "[INFO] LABEL_DATASET chưa tồn tại. Đang tạo..."

  labelohlcv "$OHLCV_DATASET" --mod vn30f1m --output "$LABEL_DATASET"

  echo "[DONE] Đã tạo LABEL_DATASET"
else
  echo "[SKIP] LABEL_DATASET đã tồn tại"
fi

# 3. Check FEATURE_DATASET
if [ ! -f "$FEATURE_DATASET" ]; then
  echo "[INFO] FEATURE_DATASET chưa tồn tại. Đang tạo..."

  autofcholv extract "$OHLCV_DATASET" --output "$FEATURE_DATASET"

  echo "[DONE] Đã tạo FEATURE_DATASET"
else
  echo "[SKIP] FEATURE_DATASET đã tồn tại"
fi

echo "=== DONE ==="
