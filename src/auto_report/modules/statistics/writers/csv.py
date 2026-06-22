"""CSV outputs for the statistics module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...base.writers.csv import write_dataframe


def write_statistics_csv(
    output_dir: Path,
    feature_stats: pd.DataFrame,
    label_stats: pd.DataFrame,
) -> list[Path]:
    return [
        write_dataframe(output_dir / "statistics_feature_column_statistics.csv", feature_stats),
        write_dataframe(output_dir / "statistics_label_column_statistics.csv", label_stats),
    ]
