# vnstock-analytics

Analytics and auto-report tooling for VN30F1M market datasets.

## Overview

This project provides two main tools:

- **CLI report pipeline** (`src/auto_report/`) — generates statistical and ML-based predictability reports from CSV feature/label datasets.
- **Jupyter notebooks** (`notebooks/VN30F1M/`) — exploratory analysis and visualization helpers.

## Project Structure

```text
datasets/              # Raw CSV inputs (git-ignored)
notebooks/VN30F1M/     # Jupyter notebooks and utils.py
reports/               # Generated report outputs (git-ignored)
scripts/               # Docker entrypoint
src/
  auto_report/         # Report pipeline package
    modules/
      statistics/      # Dataset statistics module
      xgboost/         # XGBoost label predictability module
    config.json        # Active run config
    config.sample.json # Config template
```

## Quick Start (local)

**Requirements:** Python 3.12–3.13, [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
cd src
uv sync

# Place datasets under datasets/
#   datasets/VN30F1M_5m.csv
#   datasets/VN30F1M_5m_features.csv
#   datasets/VN30F1M_5m_labels.csv

# Run all modules
src/.venv/bin/python src/auto_report.py

# Run with a custom config
src/.venv/bin/python src/auto_report.py --config /path/to/config.json
```

Or via the installed script (after `uv sync`):

```bash
cd src && uv run auto-report
```

## Quick Start (Docker / JupyterLab)

```bash
./start-docker.sh
```

Opens JupyterLab at [http://localhost:8888](http://localhost:8888).  
Notebooks and datasets are mounted as volumes — changes persist on the host.

## Configuration

Copy `src/auto_report/config.sample.json` and edit as needed:

```json
{
  "module": "all",
  "data_dir": "datasets",
  "output_dir": "reports",
  "raw_file": "VN30F1M_5m.csv",
  "features_file": "VN30F1M_5m_features.csv",
  "labels_file": "VN30F1M_5m_labels.csv",
  "modules": {
    "statistics": { "top_n": 30 },
    "xgboost": { "targets": "all", "test_size": 0.15, "val_size": 0.15 }
  }
}
```

`"module"` accepts `"all"`, `"statistics"`, or `"xgboost"`.

## Report Outputs

| File | Module | Description |
|------|--------|-------------|
| `eda_report.html` / `.md` | eda | EDA summary: distributions, correlations, label balance |
| `eda_feature_analysis.csv` | eda | Per-feature skewness, kurtosis, outlier rate |
| `eda_correlation_matrix.csv` | eda | Pearson correlation matrix (numeric features) |
| `eda_top_correlations.csv` | eda | High-correlation feature pairs |
| `eda_label_distribution.csv` | eda | Per-label class counts and imbalance ratio |
| `eda_correlation_heatmap.png` | eda | Correlation heatmap chart |
| `statistics_report.html` / `.md` | statistics | Feature and label column statistics |
| `statistics_feature_column_statistics.csv` | statistics | Per-feature stats (missing rate, distribution) |
| `statistics_label_column_statistics.csv` | statistics | Per-label stats |
| `xgboost_report.html` / `.md` | xgboost | Label predictability summary |
| `xgboost_dataset_summary.json` | xgboost | Dataset and split metadata |
| `xgboost_label_metrics.csv` | xgboost | Per-label model metrics |
| `xgboost_feature_importance_by_label.csv` | xgboost | Feature importance per label |
| `confusion_matrix_<model>.png` | xgboost | Confusion matrices |

## Dependencies

Key packages (see `src/pyproject.toml` for full list):

- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `scipy`
- `labelohlcv` — OHLCV label generation
- `autofcholv` — feature engineering helpers
- `jupyterlab` — notebook interface
