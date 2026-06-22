"""Markdown output for the XGBoost module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ....reporting import markdown_table
from ...base.writers.markdown import write_markdown


def write_xgboost_markdown(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    target_names: list[str],
    args: Any,
) -> Path:
    top_importance = (
        importance_df.sort_values(["label", "rank"])
        .groupby("label", group_keys=False)
        .head(args.top_n)
    )
    report = f"""# Multi-Label XGBoost Feature Importance

## Run
- data_dir: `{args.data_dir}`
- targets: {len(target_names)} ({", ".join(target_names)})
- max_rows: {args.max_rows or "all"}
- xgboost: n_estimators={args.xgb_n_estimators}, max_depth={args.xgb_max_depth}, learning_rate={args.xgb_learning_rate}
- xgboost_device_requested: {"cuda" if args.gpu else "cpu"}

## Label Metrics
{markdown_table(metrics_df.sort_values("primary_metric", ascending=False), ["label", "task_type", "primary_metric_name", "primary_metric", "baseline_metric", "metric_lift", "rows_total", "top_features"], args.top_n)}

## Top Feature Importance Per Label
{markdown_table(top_importance, ["label", "task_type", "rank", "feature", "importance"], args.top_n * max(1, len(target_names)))}

## Files
- `xgboost_report.html`
- `xgboost_dataset_summary.json`
- `xgboost_label_metrics.csv`
- `xgboost_feature_importance_by_label.csv`
- `xgboost_feature_importance_matrix.csv`
- `xgboost_feature_importance_heatmap.png`
- `xgboost_classification_macro_f1.png`
- `xgboost_regression_r2.png`
- `xgboost_top_features_overall.png`
"""
    return write_markdown(output_dir / "xgboost_report.md", report)
