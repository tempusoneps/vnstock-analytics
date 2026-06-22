"""CSV outputs for the EDA module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...base.writers.csv import write_dataframe


def write_eda_csv(
    output_dir: Path,
    feature_analysis: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    top_correlations: pd.DataFrame,
    label_distribution: pd.DataFrame,
) -> list[Path]:
    files = [
        write_dataframe(output_dir / "eda_feature_analysis.csv", feature_analysis),
        write_dataframe(output_dir / "eda_label_distribution.csv", label_distribution),
    ]
    if not corr_matrix.empty:
        files.append(write_dataframe(output_dir / "eda_correlation_matrix.csv", corr_matrix, index=True))
    if not top_correlations.empty:
        files.append(write_dataframe(output_dir / "eda_top_correlations.csv", top_correlations))
    return files
