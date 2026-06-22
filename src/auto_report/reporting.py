"""Console, JSON, CSV, and markdown reporting helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import DatasetBundle, FeatureSpec, SplitData


def date_range(df: pd.DataFrame) -> dict[str, str]:
    dates = pd.to_datetime(df["Date"], errors="coerce")
    return {
        "start": "" if dates.isna().all() else str(dates.min()),
        "end": "" if dates.isna().all() else str(dates.max()),
    }


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int) -> str:
    if df.empty:
        return "_No rows._"

    shown = df.loc[:, [column for column in columns if column in df.columns]].head(max_rows).copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def describe_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_rows = len(features)

    for column in features.columns:
        series = features[column]
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        missing_rate = 0.0 if total_rows == 0 else float(missing_count / total_rows)
        unique_count = int(series.nunique(dropna=True))

        row: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "rows": int(total_rows),
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "unique_count": unique_count,
            "numeric_mean": pd.NA,
            "numeric_std": pd.NA,
            "numeric_min": pd.NA,
            "numeric_p25": pd.NA,
            "numeric_median": pd.NA,
            "numeric_p75": pd.NA,
            "numeric_max": pd.NA,
            "top_value": "",
            "top_value_count": pd.NA,
            "top_value_rate": pd.NA,
        }

        if pd.api.types.is_bool_dtype(series):
            top_values = series.dropna().astype(str).value_counts()
            if not top_values.empty:
                top_count = int(top_values.iloc[0])
                row.update(
                    {
                        "top_value": top_values.index[0],
                        "top_value_count": top_count,
                        "top_value_rate": 0.0 if total_rows == 0 else float(top_count / total_rows),
                    }
                )
        elif pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
            row.update(
                {
                    "numeric_mean": numeric.mean(),
                    "numeric_std": numeric.std(),
                    "numeric_min": numeric.min(),
                    "numeric_p25": numeric.quantile(0.25),
                    "numeric_median": numeric.median(),
                    "numeric_p75": numeric.quantile(0.75),
                    "numeric_max": numeric.max(),
                }
            )
        else:
            top_values = series.dropna().astype(str).value_counts()
            if not top_values.empty:
                top_count = int(top_values.iloc[0])
                row.update(
                    {
                        "top_value": top_values.index[0],
                        "top_value_count": top_count,
                        "top_value_rate": 0.0 if total_rows == 0 else float(top_count / total_rows),
                    }
                )

        rows.append(row)

    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats
    return stats.sort_values(["missing_rate", "unique_count", "column"], ascending=[False, False, True])


def write_dataset_summary(
    output_dir: Path,
    bundle: DatasetBundle,
    labeled_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    target: str,
    split: SplitData,
) -> dict[str, Any]:
    summary = {
        "merge": bundle.merge_summary,
        "target": target,
        "total_merged_rows": int(len(bundle.merged)),
        "labeled_rows": int(len(labeled_df)),
        "target_distribution": labeled_df[target].value_counts(dropna=False).to_dict(),
        "date_range": date_range(labeled_df),
        "features": {
            "total_used": len(feature_spec.feature_columns),
            "numeric": len(feature_spec.numeric_columns),
            "categorical": len(feature_spec.categorical_columns),
            "excluded_leakage_columns": feature_spec.leakage_columns,
            "dropped_all_missing_columns": feature_spec.dropped_columns,
        },
        "split": {
            "train_rows": int(len(split.train_idx)),
            "validation_rows": int(len(split.val_idx)),
            "test_rows": int(len(split.test_idx)),
            "class_names": split.class_names,
        },
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def print_terminal_summary(
    bundle: DatasetBundle,
    labeled_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    label_columns: list[str],
    target: str,
    split: SplitData,
    max_rows: int,
) -> None:
    source_feature_count = len([column for column in bundle.features.columns if column != "Date"])
    label_count = len(label_columns)
    target_counts = labeled_df[target].value_counts(dropna=False)
    max_rows_note = f" after --max-rows={max_rows}" if max_rows > 0 else ""

    print("\n=== DATA SUMMARY ===")
    print(f"Feature file rows      : {len(bundle.features):,}")
    print(f"Label file rows        : {len(bundle.labels):,}")
    print(f"Merged rows            : {len(bundle.merged):,}")
    print(f"Labeled target rows    : {len(labeled_df):,}{max_rows_note}")
    print(f"Feature columns source : {source_feature_count:,}")
    print(
        "Feature columns used   : "
        f"{len(feature_spec.feature_columns):,} "
        f"({len(feature_spec.numeric_columns):,} numeric, "
        f"{len(feature_spec.categorical_columns):,} categorical)"
    )
    print(f"Label columns          : {label_count:,}")
    print(f"Target column          : {target}")
    print(f"Target classes         : {len(split.class_names):,}")
    print("Target distribution:")
    for label in split.class_names:
        print(f"  - {label}: {int(target_counts.get(label, 0)):,}")
    unknown_labels = [label for label in target_counts.index if label not in split.class_names]
    for label in unknown_labels:
        print(f"  - {label}: {int(target_counts.get(label, 0)):,}")
    if feature_spec.leakage_columns:
        print(f"Excluded leakage cols  : {', '.join(feature_spec.leakage_columns)}")
    print("====================\n")


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    metrics_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    importance_frames: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    best_metric = pd.DataFrame()
    if not metrics_df.empty:
        best_metric = metrics_df[metrics_df["split"] == "test"].sort_values("macro_f1", ascending=False)

    importance_section = "_Permutation importance skipped or unavailable._"
    if importance_frames:
        sections = []
        for model_name, frame in importance_frames.items():
            sections.append(f"### {model_name}\n")
            sections.append(
                markdown_table(
                    frame,
                    ["feature", "importance_mean", "importance_std", "rows_used", "repeats"],
                    args.top_n,
                )
            )
        importance_section = "\n\n".join(sections)

    report = f"""# Feature Label Predictability Report

## Run
- data_dir: `{args.data_dir}`
- target: `{args.target}`
- rows: merged={summary["total_merged_rows"]}, labeled={summary["labeled_rows"]}
- date_range: {summary["date_range"]["start"]} to {summary["date_range"]["end"]}
- split: train={summary["split"]["train_rows"]}, validation={summary["split"]["validation_rows"]}, test={summary["split"]["test_rows"]}
- classes: {", ".join(summary["split"]["class_names"])}
- excluded leakage columns: {", ".join(summary["features"]["excluded_leakage_columns"]) or "none"}

## Target Distribution
```json
{json.dumps(summary["target_distribution"], indent=2)}
```

## Test Metrics
{markdown_table(best_metric, ["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss"], args.top_n)}

## Top Numeric Relationships
{markdown_table(numeric_df, ["feature", "mutual_info", "f_score", "f_pvalue", "kruskal_stat", "missing_rate"], args.top_n)}

## Top Categorical Relationships
{markdown_table(categorical_df, ["feature", "cramers_v", "chi2", "pvalue", "missing_rate", "unique_count"], args.top_n)}

## Permutation Importance
{importance_section}

## Files
- `dataset_summary.json`
- `feature_relationship_numeric.csv`
- `feature_relationship_categorical.csv`
- `categorical_lift_top.csv`
- `metrics.csv`
- `classification_report_<model>.csv`
- `confusion_matrix_<model>.png`
- `permutation_importance_<model>.csv`
"""
    (output_dir / "report.md").write_text(report)
