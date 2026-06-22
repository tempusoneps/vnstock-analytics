"""Markdown output for the EDA module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ....reporting import markdown_table
from ...base.writers.markdown import write_markdown


def write_eda_markdown(
    output_dir: Path,
    feature_analysis: pd.DataFrame,
    top_correlations: pd.DataFrame,
    label_distribution: pd.DataFrame,
    top_n: int,
) -> Path:
    constant_cols = (
        feature_analysis.loc[feature_analysis["is_constant"], "column"].tolist()
        if not feature_analysis.empty
        else []
    )
    constant_note = ", ".join(f"`{c}`" for c in constant_cols) if constant_cols else "_none_"
    corr_section = (
        markdown_table(top_correlations, ["feature_a", "feature_b", "correlation"], top_n)
        if not top_correlations.empty
        else "_No pairs above threshold._"
    )

    report = f"""# EDA Report

## Feature Analysis
{markdown_table(
    feature_analysis,
    ["column", "dtype", "missing_rate", "unique_count", "skewness", "kurtosis", "outlier_count", "outlier_rate"],
    top_n,
)}

**Constant columns (zero variance):** {constant_note}

## High-Correlation Feature Pairs
{corr_section}

## Label Distribution
{markdown_table(label_distribution, ["label", "class", "count", "rate", "imbalance_ratio", "is_imbalanced"], top_n)}

## Files
- `eda_feature_analysis.csv`
- `eda_correlation_matrix.csv`
- `eda_top_correlations.csv`
- `eda_label_distribution.csv`
- `eda_correlation_heatmap.png`
- `eda_report.html`
"""
    return write_markdown(output_dir / "eda_report.md", report)
