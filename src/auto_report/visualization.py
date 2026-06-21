"""Static visualizations and HTML report for multi-label XGBoost analysis."""

from __future__ import annotations

import html
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_empty_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_importance_heatmap(
    importance_df: pd.DataFrame,
    output_dir: Path,
    top_n_features: int,
) -> Path:
    path = output_dir / "feature_importance_heatmap.png"
    if importance_df.empty:
        _save_empty_plot(path, "Feature Importance Heatmap", "No feature importance rows.")
        return path

    top_features = (
        importance_df.groupby("feature")["importance"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_features)
        .index
    )
    matrix = (
        importance_df[importance_df["feature"].isin(top_features)]
        .pivot_table(index="label", columns="feature", values="importance", aggfunc="sum", fill_value=0.0)
        .reindex(columns=top_features)
    )

    fig_width = max(10, min(24, 0.45 * len(matrix.columns)))
    fig_height = max(5, min(16, 0.45 * len(matrix.index)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label="Normalized XGBoost gain")
    ax.set_title("Feature Importance by Label")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Label")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=55, ha="right")
    ax.set_yticklabels(matrix.index)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_metric_bars(metrics_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []

    classification = metrics_df[metrics_df["task_type"] == "classification"].copy()
    classification_path = output_dir / "classification_macro_f1.png"
    if classification.empty:
        _save_empty_plot(classification_path, "Classification Predictability", "No classification targets.")
    else:
        classification = classification.sort_values("primary_metric", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(classification))))
        ax.barh(classification["label"], classification["baseline_metric"], color="#c9d1d9", label="baseline")
        ax.barh(classification["label"], classification["primary_metric"], color="#2f80ed", alpha=0.82, label="xgboost")
        ax.set_title("Classification Targets - Macro F1")
        ax.set_xlabel("Macro F1")
        ax.legend()
        fig.tight_layout()
        fig.savefig(classification_path, dpi=160)
        plt.close(fig)
    paths.append(classification_path)

    regression = metrics_df[metrics_df["task_type"] == "regression"].copy()
    regression_path = output_dir / "regression_r2.png"
    if regression.empty:
        _save_empty_plot(regression_path, "Regression Predictability", "No regression targets.")
    else:
        regression = regression.sort_values("primary_metric", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(regression))))
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.barh(regression["label"], regression["primary_metric"], color="#27ae60", alpha=0.82, label="xgboost")
        ax.set_title("Regression Targets - R2")
        ax.set_xlabel("R2")
        ax.legend()
        fig.tight_layout()
        fig.savefig(regression_path, dpi=160)
        plt.close(fig)
    paths.append(regression_path)
    return paths


def save_top_features_bar(
    importance_df: pd.DataFrame,
    output_dir: Path,
    top_n_features: int,
) -> Path:
    path = output_dir / "top_features_overall.png"
    if importance_df.empty:
        _save_empty_plot(path, "Top Features Overall", "No feature importance rows.")
        return path

    top = (
        importance_df.groupby("feature")["importance"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_features)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(top))))
    ax.barh(top.index, top.values, color="#9b51e0", alpha=0.82)
    ax.set_title("Top Features Across All Labels")
    ax.set_xlabel("Sum of normalized XGBoost gain")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_html_report(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    image_paths: list[Path],
    top_n: int,
) -> None:
    top_metrics = metrics_df.sort_values("primary_metric", ascending=False).copy()
    top_importance = (
        importance_df.sort_values(["label", "rank"])
        .groupby("label", group_keys=False)
        .head(top_n)
        .copy()
    )

    image_html = "\n".join(
        f'<section><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"></section>'
        for path in image_paths
    )
    metrics_html = top_metrics.to_html(index=False, classes="table", float_format=lambda value: f"{value:.6g}")
    importance_html = top_importance.to_html(index=False, classes="table", float_format=lambda value: f"{value:.6g}")

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Feature Label XGBoost Importance Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #17202a;
      background: #ffffff;
    }}
    h1, h2 {{ margin-bottom: 8px; }}
    section {{ margin: 24px 0; }}
    img {{
      display: block;
      max-width: 100%;
      border: 1px solid #d8dee4;
    }}
    .table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    .table th, .table td {{
      border: 1px solid #d8dee4;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    .table th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>Feature Label XGBoost Importance Report</h1>
  <p>Importance is normalized XGBoost gain aggregated back to original feature names.</p>
  <h2>Visualizations</h2>
  {image_html}
  <h2>Label Metrics</h2>
  {metrics_html}
  <h2>Top Feature Importance Per Label</h2>
  {importance_html}
</body>
</html>
"""
    (output_dir / "report.html").write_text(document)


def write_multi_label_visualizations(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int,
) -> None:
    image_paths = [
        save_importance_heatmap(importance_df, output_dir, top_n),
        *save_metric_bars(metrics_df, output_dir),
        save_top_features_bar(importance_df, output_dir, top_n),
    ]
    write_html_report(output_dir, metrics_df, importance_df, image_paths, top_n)
