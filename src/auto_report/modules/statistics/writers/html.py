"""HTML output for the statistics module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...base.writers.html import dataframe_html, write_html


def write_statistics_html(
    output_dir: Path,
    feature_stats: pd.DataFrame,
    label_stats: pd.DataFrame,
    top_n: int,
) -> Path:
    feature_columns = [
        "column",
        "dtype",
        "missing_rate",
        "unique_count",
        "numeric_mean",
        "numeric_std",
        "numeric_min",
        "numeric_median",
        "numeric_max",
        "top_value",
        "top_value_rate",
    ]
    label_columns = ["column", "dtype", "missing_rate", "unique_count", "top_value", "top_value_rate"]
    feature_html = dataframe_html(feature_stats[[column for column in feature_columns if column in feature_stats.columns]].head(top_n))
    label_html = dataframe_html(label_stats[[column for column in label_columns if column in label_stats.columns]].head(top_n))

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dataset Statistics Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; background: #ffffff; }}
    .table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    .table th, .table td {{ border: 1px solid #d8dee4; padding: 6px 8px; text-align: left; vertical-align: top; }}
    .table th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>Dataset Statistics Report</h1>
  <h2>Feature Column Statistics</h2>
  {feature_html}
  <h2>Label Column Statistics</h2>
  {label_html}
</body>
</html>
"""
    return write_html(output_dir / "statistics_report.html", document)
