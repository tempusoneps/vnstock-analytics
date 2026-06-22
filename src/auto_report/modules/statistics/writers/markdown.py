"""Markdown output for the statistics module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ....reporting import markdown_table
from ...base.writers.markdown import write_markdown


def write_statistics_markdown(
    output_dir: Path,
    feature_stats: pd.DataFrame,
    label_stats: pd.DataFrame,
    top_n: int,
) -> Path:
    report = f"""# Dataset Statistics Report

## Feature Column Statistics
{markdown_table(feature_stats, ["column", "dtype", "missing_rate", "unique_count", "numeric_mean", "numeric_std", "numeric_min", "numeric_median", "numeric_max", "top_value", "top_value_rate"], top_n)}

## Label Column Statistics
{markdown_table(label_stats, ["column", "dtype", "missing_rate", "unique_count", "top_value", "top_value_rate"], top_n)}

## Files
- `statistics_feature_column_statistics.csv`
- `statistics_label_column_statistics.csv`
- `statistics_report.html`
"""
    return write_markdown(output_dir / "statistics_report.md", report)
