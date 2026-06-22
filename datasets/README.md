# Datasets

This folder contains the input datasets used by `auto_report`.

To prepare a complete VN30F1M dataset, create these files:

```text
VN30F1M_5m.csv
VN30F1M_5m_features.csv
VN30F1M_5m_labels.csv
```

## 1. Download Raw Dataset

Download the raw OHLCV dataset and place it here:

```text
datasets/VN30F1M_5m.csv
```

Optionally download it from GitHub:

[VN30F1M_5m.csv](https://github.com/tempusoneps/vn-stock-data/blob/main/VN30F1M/data_ohlcv/VN30F1M_5m.csv)

From the `datasets/` directory, you can run:

```bash
curl -L \
  https://raw.githubusercontent.com/tempusoneps/vn-stock-data/main/VN30F1M/data_ohlcv/VN30F1M_5m.csv \
  -o VN30F1M_5m.csv
```

The raw file is the source for both feature extraction and label generation.

## 2. Create Feature Dataset

From the `datasets/` directory, run:

```bash
autofcholv extract VN30F1M_5m.csv --output VN30F1M_5m_features.csv
```

Expected output:

```text
datasets/VN30F1M_5m_features.csv
```

## 3. Create Label Dataset

From the `datasets/` directory, run:

```bash
labelohlcv VN30F1M_5m.csv --mod vn30f1m --output VN30F1M_5m_labels.csv
```

Expected output:

```text
datasets/VN30F1M_5m_labels.csv
```

## Complete Dataset

After these steps, `auto_report` can merge the feature and label datasets using the shared `Date` column.

The generated analytics-ready file may also be created by notebook helpers:

```text
datasets/VN30F1M_5m_ready.csv
```
