"""HTML and image outputs for the XGBoost module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ....visualization import write_multi_label_visualizations


def write_xgboost_html(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int,
) -> list[Path]:
    write_multi_label_visualizations(output_dir, metrics_df, importance_df, top_n, prefix="xgboost")
    return [
        output_dir / "xgboost_report.html",
        output_dir / "xgboost_feature_importance_heatmap.png",
        output_dir / "xgboost_classification_macro_f1.png",
        output_dir / "xgboost_regression_r2.png",
        output_dir / "xgboost_top_features_overall.png",
    ]
