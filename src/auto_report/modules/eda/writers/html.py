"""HTML output for the EDA module."""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

from ...base.writers.html import dataframe_html, write_html


def _table_section(df: pd.DataFrame, columns: list[str], top_n: int) -> str:
    if df.empty:
        return "<p><em>No data.</em></p>"
    cols = [c for c in columns if c in df.columns]
    return f'<div class="table-wrap">{dataframe_html(df[cols].head(top_n))}</div>'


def write_eda_html(
    output_dir: Path,
    feature_analysis: pd.DataFrame,
    top_correlations: pd.DataFrame,
    label_distribution: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    top_n: int,
    heatmap_path: "Path | None",
) -> Path:
    feature_html = _table_section(
        feature_analysis,
        ["column", "dtype", "missing_rate", "unique_count", "is_constant", "skewness", "kurtosis", "outlier_count", "outlier_rate"],
        top_n,
    )
    corr_html = (
        _table_section(top_correlations, ["feature_a", "feature_b", "correlation"], top_n)
        if not top_correlations.empty
        else "<p><em>No pairs above threshold.</em></p>"
    )
    label_html = _table_section(
        label_distribution,
        ["label", "class", "count", "rate", "imbalance_ratio", "is_imbalanced"],
        top_n,
    )

    heatmap_html = ""
    if heatmap_path and heatmap_path.exists():
        encoded = base64.b64encode(heatmap_path.read_bytes()).decode()
        heatmap_html = (
            f'<img src="data:image/png;base64,{encoded}" '
            'style="max-width:100%;height:auto;" alt="Correlation Heatmap">'
        )

    constant_count = int(feature_analysis["is_constant"].sum()) if not feature_analysis.empty else 0
    label_count = int(label_distribution["label"].nunique()) if not label_distribution.empty else 0
    imbalanced_count = (
        int(label_distribution.drop_duplicates("label")["is_imbalanced"].sum())
        if not label_distribution.empty
        else 0
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EDA Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #17202a; background: #ffffff; }}
    h1 {{ color: #1a5276; }}
    h2 {{ color: #21618c; border-bottom: 1px solid #d8dee4; padding-bottom: 4px; margin-top: 32px; }}
    .table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    .table th, .table td {{ border: 1px solid #d8dee4; padding: 5px 8px; text-align: left; vertical-align: top; white-space: nowrap; }}
    .table th {{ background: #f6f8fa; position: sticky; top: 0; }}
    .table-wrap {{ overflow-x: auto; margin-bottom: 20px; }}
    .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
    .card {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 14px 22px; min-width: 130px; text-align: center; }}
    .card .val {{ font-size: 30px; font-weight: bold; color: #1a5276; }}
    .card .lbl {{ font-size: 12px; color: #5d6d7e; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>EDA Report</h1>

  <div class="cards">
    <div class="card"><div class="val">{len(feature_analysis)}</div><div class="lbl">Features</div></div>
    <div class="card"><div class="val">{constant_count}</div><div class="lbl">Constant Cols</div></div>
    <div class="card"><div class="val">{len(top_correlations)}</div><div class="lbl">High-Corr Pairs</div></div>
    <div class="card"><div class="val">{label_count}</div><div class="lbl">Labels</div></div>
    <div class="card"><div class="val">{imbalanced_count}</div><div class="lbl">Imbalanced Labels</div></div>
  </div>

  <h2>Feature Analysis (top {top_n})</h2>
  {feature_html}

  <h2>High-Correlation Feature Pairs</h2>
  {corr_html}

  <h2>Correlation Heatmap</h2>
  {heatmap_html if heatmap_html else "<p><em>Heatmap not available (no numeric features).</em></p>"}

  <h2>Label Distribution</h2>
  {label_html}
</body>
</html>
"""
    return write_html(output_dir / "eda_report.html", document)
