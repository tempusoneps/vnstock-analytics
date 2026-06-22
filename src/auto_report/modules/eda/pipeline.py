"""Pipeline for EDA (Exploratory Data Analysis) reports."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..base.pipeline import PipelineContext, PipelineResult
from .reports.correlation_analysis import build_correlation_matrix, build_top_correlations
from .reports.feature_analysis import build_feature_analysis
from .reports.label_analysis import build_label_distribution
from .writers.csv import write_eda_csv
from .writers.html import write_eda_html
from .writers.markdown import write_eda_markdown


def _save_correlation_heatmap(corr_matrix: pd.DataFrame, output_dir) -> "Path | None":
    from pathlib import Path

    if corr_matrix.empty:
        return None

    n = len(corr_matrix)
    size = max(8, min(n * 0.4, 24))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    data = corr_matrix.values.astype(float)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = corr_matrix.columns.tolist()
    fontsize = max(5, min(9, 180 // n))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_title("Feature Correlation Matrix (Pearson)", fontsize=11)

    fig.tight_layout()
    path = Path(output_dir) / "eda_correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


class EdaPipeline:
    name = "eda"

    def run(self, context: PipelineContext) -> PipelineResult:
        config = context.config
        bundle = context.bundle
        output_dir = context.output_dir

        print("[INFO] EDA: analysing feature distributions and outliers")
        feature_analysis = build_feature_analysis(bundle.features, outlier_iqr_factor=config.eda_outlier_iqr_factor)

        print("[INFO] EDA: computing feature correlation matrix")
        corr_matrix = build_correlation_matrix(bundle.features, max_features=config.eda_max_corr_features)
        top_correlations = build_top_correlations(corr_matrix, threshold=config.eda_corr_threshold)

        print("[INFO] EDA: analysing label distributions")
        label_distribution = build_label_distribution(bundle.labels)

        output_files = []

        heatmap_path = _save_correlation_heatmap(corr_matrix, output_dir)
        if heatmap_path:
            output_files.append(heatmap_path)

        output_files.extend(
            write_eda_csv(output_dir, feature_analysis, corr_matrix, top_correlations, label_distribution)
        )
        output_files.append(
            write_eda_markdown(output_dir, feature_analysis, top_correlations, label_distribution, config.top_n)
        )
        output_files.append(
            write_eda_html(
                output_dir,
                feature_analysis,
                top_correlations,
                label_distribution,
                corr_matrix,
                config.top_n,
                heatmap_path,
            )
        )

        constant_count = int(feature_analysis["is_constant"].sum()) if not feature_analysis.empty else 0
        label_count = int(label_distribution["label"].nunique()) if not label_distribution.empty else 0
        imbalanced_count = (
            int(label_distribution.drop_duplicates("label")["is_imbalanced"].sum())
            if not label_distribution.empty
            else 0
        )

        summary = {
            "feature_columns_analyzed": len(feature_analysis),
            "constant_columns": constant_count,
            "high_correlation_pairs": len(top_correlations),
            "corr_threshold": config.eda_corr_threshold,
            "outlier_iqr_factor": config.eda_outlier_iqr_factor,
            "label_columns_analyzed": label_count,
            "imbalanced_labels": imbalanced_count,
        }
        return PipelineResult(name=self.name, output_files=output_files, summary=summary)
