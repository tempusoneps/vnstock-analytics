#!/bin/bash

set -e  # exit nếu có lỗi

OHCLV_DATASET="/algo_analytics/vn-stock-data/VN30F1M/data_ohlcv/VN30F1M_5m.csv"
ANALYTICS_DATASET="/algo_analytics/phantich-chungkhoan/VN30F1M/VN30F1M_5m_features.csv"
AI_DATASET="/algo_analytics/ai-chungkhoan/VN30F1M/VN30F1M_5m_features.csv"

echo "=== START DATA PREP ==="

# 1. Check ANALYTICS_DATASET
if [ ! -f "$ANALYTICS_DATASET" ]; then
  echo "[INFO] ANALYTICS_DATASET chưa tồn tại. Đang tạo..."

  autofcholv extract "$OHCLV_DATASET" --output "$ANALYTICS_DATASET"

  echo "[DONE] Đã tạo ANALYTICS_DATASET"
else
  echo "[SKIP] ANALYTICS_DATASET đã tồn tại"
fi

# 2. Check AI_DATASET
if [ ! -f "$AI_DATASET" ]; then
  echo "[INFO] AI_DATASET chưa tồn tại. Đang copy từ ANALYTICS_DATASET..."

  # đảm bảo thư mục tồn tại
  mkdir -p "$(dirname "$AI_DATASET")"

  cp "$ANALYTICS_DATASET" "$AI_DATASET"

  echo "[DONE] Đã tạo AI_DATASET"
else
  echo "[SKIP] AI_DATASET đã tồn tại"
fi

echo "=== DONE ==="

echo "=== START JUPYTER LAB ==="

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=''