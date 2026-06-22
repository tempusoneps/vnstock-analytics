"""CSV outputs for the XGBoost module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...base.writers.csv import write_dataframe


def write_xgboost_csv(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    importance_matrix: pd.DataFrame,
) -> list[Path]:
    return [
        write_dataframe(output_dir / "xgboost_label_metrics.csv", metrics_df),
        write_dataframe(output_dir / "xgboost_feature_importance_by_label.csv", importance_df),
        write_dataframe(output_dir / "xgboost_feature_importance_matrix.csv", importance_matrix, index=True),
    ]
