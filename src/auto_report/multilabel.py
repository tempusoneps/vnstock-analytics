"""Multi-label XGBoost training and feature-importance workflow."""

from __future__ import annotations

import re
import warnings
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .constants import DEFAULT_CLASS_ORDER, KEY_COLUMNS
from .data import load_and_merge, normalize_target, resolve_data_dir
from .features import build_feature_spec, coerce_feature_columns
from .modeling import build_preprocessor
from .reporting import date_range, markdown_table
from .utils import fail, normalize_path
from .visualization import write_multi_label_visualizations


@dataclass
class TargetResult:
    metrics: dict[str, Any]
    importance: pd.DataFrame


def parse_target_names(targets_arg: str, target_alias: str | None, label_columns: list[str]) -> list[str]:
    raw_targets = target_alias or targets_arg
    if raw_targets.strip().lower() == "all":
        return list(label_columns)

    requested = [target.strip() for target in raw_targets.split(",") if target.strip()]
    if not requested:
        fail("No targets selected.")

    missing = [target for target in requested if target not in label_columns]
    if missing:
        fail(f"Unknown target columns: {missing}. Available labels: {label_columns}")
    return requested


def infer_task_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "regression"

    values = series.astype("string").str.strip()
    converted = pd.to_numeric(values, errors="coerce")
    non_missing = int(values.notna().sum())
    numeric_ratio = 0.0 if non_missing == 0 else float(converted.notna().sum() / non_missing)
    unique_count = int(converted.dropna().nunique())

    if numeric_ratio >= 0.95 and unique_count > 12:
        return "regression"
    return "classification"


def split_indices(n_rows: int, val_size: float, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < val_size < 0.5:
        fail("--val-size must be between 0 and 0.5")
    if not 0 < test_size < 0.5:
        fail("--test-size must be between 0 and 0.5")
    if val_size + test_size >= 0.8:
        fail("--val-size + --test-size must be below 0.8")

    test_count = max(1, int(round(n_rows * test_size)))
    val_count = max(1, int(round(n_rows * val_size)))
    train_count = n_rows - val_count - test_count
    if train_count < 20:
        fail(f"Not enough rows for time split: rows={n_rows}, train={train_count}")
    return (
        np.arange(0, train_count),
        np.arange(train_count, train_count + val_count),
        np.arange(train_count + val_count, n_rows),
    )


def class_order(target: str, values: pd.Series) -> list[str]:
    observed = set(values.dropna().astype(str).unique())
    ordered = [value for value in DEFAULT_CLASS_ORDER if value in observed]
    ordered.extend(sorted(value for value in observed if value not in ordered))
    return ordered


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "target"


def xgb_classifier(class_count: int, args: Any):
    from xgboost import XGBClassifier

    device = "cuda" if args.gpu else "cpu"
    return XGBClassifier(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=class_count,
        tree_method="hist",
        device=device,
        importance_type="gain",
        random_state=args.random_state,
        n_jobs=-1,
    )


def xgb_regressor(args: Any):
    from xgboost import XGBRegressor

    device = "cuda" if args.gpu else "cpu"
    return XGBRegressor(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device=device,
        importance_type="gain",
        random_state=args.random_state,
        n_jobs=-1,
    )


def map_encoded_feature_name(encoded_name: str, numeric_columns: list[str], categorical_columns: list[str]) -> str:
    if "__" in encoded_name:
        _, raw_name = encoded_name.split("__", 1)
    else:
        raw_name = encoded_name

    if raw_name in numeric_columns:
        return raw_name

    for column in sorted(categorical_columns, key=len, reverse=True):
        if raw_name == column or raw_name.startswith(f"{column}_"):
            return column
    return raw_name


def xgboost_importance(
    target: str,
    task_type: str,
    model: Pipeline,
    numeric_columns: list[str],
    categorical_columns: list[str],
    top_n: int,
) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame()

    encoded_names = preprocessor.get_feature_names_out()
    rows = []
    for encoded_name, importance in zip(encoded_names, importances):
        feature = map_encoded_feature_name(encoded_name, numeric_columns, categorical_columns)
        rows.append(
            {
                "label": target,
                "task_type": task_type,
                "feature": feature,
                "encoded_feature": encoded_name,
                "encoded_importance": float(importance),
            }
        )

    encoded_df = pd.DataFrame(rows)
    if encoded_df.empty:
        return encoded_df

    grouped = (
        encoded_df.groupby(["label", "task_type", "feature"], as_index=False)["encoded_importance"]
        .sum()
        .rename(columns={"encoded_importance": "importance"})
    )
    total = float(grouped["importance"].sum())
    if total > 0:
        grouped["importance"] = grouped["importance"] / total
    grouped = grouped.sort_values("importance", ascending=False).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped["is_top_n"] = grouped["rank"] <= top_n
    return grouped


def evaluate_classification(
    target: str,
    model: Pipeline,
    baseline: DummyClassifier,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
    row_counts: dict[str, int],
) -> dict[str, Any]:
    model_pred = model.predict(x_test)
    baseline_pred = baseline.predict(x_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        macro_f1 = f1_score(y_test, model_pred, average="macro", zero_division=0)
        baseline_macro_f1 = f1_score(y_test, baseline_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_test, model_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, model_pred)
    return {
        "label": target,
        "task_type": "classification",
        "rows_total": row_counts["total"],
        "rows_train": row_counts["train"],
        "rows_validation": row_counts["validation"],
        "rows_test": row_counts["test"],
        "rows_test_scored": len(y_test),
        "rows_test_unseen_class": row_counts.get("test_unseen_class", 0),
        "classes": ", ".join(class_names),
        "class_count": len(class_names),
        "primary_metric": macro_f1,
        "primary_metric_name": "macro_f1",
        "baseline_metric": baseline_macro_f1,
        "metric_lift": macro_f1 - baseline_macro_f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "baseline_macro_f1": baseline_macro_f1,
        "r2": np.nan,
        "baseline_r2": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
    }


def evaluate_regression(
    target: str,
    model: Pipeline,
    baseline: DummyRegressor,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    row_counts: dict[str, int],
) -> dict[str, Any]:
    model_pred = model.predict(x_test)
    baseline_pred = baseline.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, model_pred)))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    r2 = r2_score(y_test, model_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)
    return {
        "label": target,
        "task_type": "regression",
        "rows_total": row_counts["total"],
        "rows_train": row_counts["train"],
        "rows_validation": row_counts["validation"],
        "rows_test": row_counts["test"],
        "rows_test_scored": len(y_test),
        "rows_test_unseen_class": 0,
        "classes": "",
        "class_count": np.nan,
        "primary_metric": r2,
        "primary_metric_name": "r2",
        "baseline_metric": baseline_r2,
        "metric_lift": r2 - baseline_r2,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "baseline_macro_f1": np.nan,
        "r2": r2,
        "baseline_r2": baseline_r2,
        "mae": mean_absolute_error(y_test, model_pred),
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
    }


def run_one_target(
    target: str,
    merged: pd.DataFrame,
    x_all: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    args: Any,
) -> TargetResult | None:
    task_type = infer_task_type(merged[target])
    target_df = merged[[target]].copy()

    if task_type == "regression":
        y = pd.to_numeric(target_df[target], errors="coerce")
        mask = y.notna()
    else:
        y = normalize_target(target_df[target])
        mask = y.notna()

    x_target = x_all.loc[mask].copy()
    y_target = y.loc[mask].copy()
    if args.max_rows > 0:
        x_target = x_target.head(args.max_rows)
        y_target = y_target.head(args.max_rows)

    if len(y_target) < args.min_rows_per_target:
        print(f"[WARN] Skipping {target}: only {len(y_target):,} usable rows.")
        return None

    train_idx, val_idx, test_idx = split_indices(len(y_target), args.val_size, args.test_size)
    train_non_missing = x_target.iloc[train_idx].notna().any(axis=0)
    target_feature_columns = train_non_missing[train_non_missing].index.tolist()
    if not target_feature_columns:
        print(f"[WARN] Skipping {target}: no usable feature columns in train split.")
        return None
    target_numeric_columns = [column for column in numeric_columns if column in target_feature_columns]
    target_categorical_columns = [column for column in categorical_columns if column in target_feature_columns]
    x_target = x_target[target_feature_columns]
    x_train = x_target.iloc[train_idx]
    x_test = x_target.iloc[test_idx]
    row_counts = {
        "total": len(y_target),
        "train": len(train_idx),
        "validation": len(val_idx),
        "test": len(test_idx),
    }

    preprocessor = build_preprocessor(target_numeric_columns, target_categorical_columns)

    if task_type == "classification":
        train_values = y_target.iloc[train_idx].astype(str)
        test_values = y_target.iloc[test_idx].astype(str)
        classes = class_order(target, train_values)
        if len(classes) < 2:
            print(f"[WARN] Skipping {target}: fewer than 2 classes.")
            return None

        test_known_mask = test_values.isin(classes).to_numpy()
        unseen_test_rows = int((~test_known_mask).sum())
        if unseen_test_rows:
            print(
                f"[WARN] Target {target}: {unseen_test_rows:,} test rows have "
                "classes unseen in train and will be excluded from scoring."
            )
        if not np.any(test_known_mask):
            print(f"[WARN] Skipping {target}: no test rows with classes seen in train.")
            return None

        x_test_scored = x_test.loc[test_values.index[test_known_mask]]
        y_test_values = test_values.loc[test_known_mask]
        row_counts["test_unseen_class"] = unseen_test_rows
        encoder = LabelEncoder()
        encoder.fit(classes)
        y_train = encoder.transform(train_values)
        y_test = encoder.transform(y_test_values)

        model = Pipeline([("preprocess", preprocessor), ("model", xgb_classifier(len(classes), args))])
        baseline = DummyClassifier(strategy="most_frequent")
        model.fit(x_train, y_train)
        baseline.fit(x_train, y_train)
        metrics = evaluate_classification(target, model, baseline, x_test_scored, y_test, classes, row_counts)
    else:
        y_numeric = y_target.astype(float).to_numpy()
        if float(np.nanstd(y_numeric)) == 0:
            print(f"[WARN] Skipping {target}: target is constant.")
            return None

        y_train = y_numeric[train_idx]
        y_test = y_numeric[test_idx]
        model = Pipeline([("preprocess", preprocessor), ("model", xgb_regressor(args))])
        baseline = DummyRegressor(strategy="mean")
        model.fit(x_train, y_train)
        baseline.fit(x_train, y_train)
        metrics = evaluate_regression(target, model, baseline, x_test, y_test, row_counts)

    importance = xgboost_importance(
        target,
        task_type,
        model,
        target_numeric_columns,
        target_categorical_columns,
        args.top_n,
    )
    metrics["top_features"] = ", ".join(importance.head(5)["feature"].tolist()) if not importance.empty else ""
    return TargetResult(metrics=metrics, importance=importance)


def write_multi_markdown(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    target_names: list[str],
    args: Any,
) -> None:
    top_importance = (
        importance_df.sort_values(["label", "rank"])
        .groupby("label", group_keys=False)
        .head(args.top_n)
    )
    report = f"""# Multi-Label XGBoost Feature Importance

## Run
- data_dir: `{args.data_dir}`
- targets: {len(target_names)} ({", ".join(target_names)})
- max_rows: {args.max_rows or "all"}
- xgboost: n_estimators={args.xgb_n_estimators}, max_depth={args.xgb_max_depth}, learning_rate={args.xgb_learning_rate}
- xgboost_device_requested: {"cuda" if args.gpu else "cpu"}

## Label Metrics
{markdown_table(metrics_df.sort_values("primary_metric", ascending=False), ["label", "task_type", "primary_metric_name", "primary_metric", "baseline_metric", "metric_lift", "rows_total", "top_features"], args.top_n)}

## Top Feature Importance Per Label
{markdown_table(top_importance, ["label", "task_type", "rank", "feature", "importance"], args.top_n * max(1, len(target_names)))}

## Files
- `report.html`
- `label_metrics.csv`
- `feature_importance_by_label.csv`
- `feature_importance_matrix.csv`
- `feature_importance_heatmap.png`
- `classification_macro_f1.png`
- `regression_r2.png`
- `top_features_overall.png`
"""
    (output_dir / "report.md").write_text(report)


def print_multi_summary(
    bundle: Any,
    feature_count_source: int,
    numeric_count: int,
    categorical_count: int,
    label_columns: list[str],
    target_names: list[str],
    metrics_df: pd.DataFrame,
    gpu: bool,
) -> None:
    print("\n=== MULTI-LABEL XGBOOST SUMMARY ===")
    print(f"Feature file rows      : {len(bundle.features):,}")
    print(f"Label file rows        : {len(bundle.labels):,}")
    print(f"Merged rows            : {len(bundle.merged):,}")
    print(f"Feature columns source : {feature_count_source:,}")
    print(f"Feature columns used   : {numeric_count + categorical_count:,} ({numeric_count:,} numeric, {categorical_count:,} categorical)")
    print(f"Label columns          : {len(label_columns):,}")
    print(f"Targets requested      : {len(target_names):,}")
    print(f"Targets completed      : {len(metrics_df):,}")
    print(f"XGBoost device request : {'cuda' if gpu else 'cpu'}")
    print(f"XGBoost CUDA build     : {bool(xgb.build_info().get('USE_CUDA'))}")
    if not metrics_df.empty:
        print("Targets by task:")
        for task_type, count in metrics_df["task_type"].value_counts().items():
            print(f"  - {task_type}: {int(count):,}")
        print("Top predictable labels:")
        for _, row in metrics_df.sort_values("primary_metric", ascending=False).head(8).iterrows():
            print(
                f"  - {row['label']} ({row['task_type']}): "
                f"{row['primary_metric_name']}={row['primary_metric']:.4f}, "
                f"lift={row['metric_lift']:.4f}"
            )
    print("===================================\n")


def run_multi_label(args: Any) -> None:
    args.data_dir = normalize_path(args.data_dir)
    args.data_dir = resolve_data_dir(args.data_dir, args.raw_file, args.features_file, args.labels_file)
    args.output_dir = normalize_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data from {args.data_dir}")
    bundle = load_and_merge(args.data_dir, args.raw_file, args.features_file, args.labels_file)
    label_columns = [column for column in bundle.labels.columns if column not in KEY_COLUMNS]
    target_names = parse_target_names(args.targets, args.target, label_columns)

    merged = bundle.merged.copy()
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.sort_values("Date").reset_index(drop=True)

    feature_spec = build_feature_spec(
        merged,
        label_columns=label_columns,
        target=target_names[0],
        include_leakage=args.include_leakage,
    )
    x_all, numeric_columns, categorical_columns = coerce_feature_columns(merged, feature_spec.feature_columns)
    feature_spec.numeric_columns = numeric_columns
    feature_spec.categorical_columns = categorical_columns

    print(f"[INFO] Training XGBoost for {len(target_names)} target label(s)")
    results: list[TargetResult] = []
    for target in target_names:
        print(f"[INFO] Target: {target}")
        result = run_one_target(target, merged, x_all, numeric_columns, categorical_columns, args)
        if result is not None:
            results.append(result)

    if not results:
        fail("No targets were completed.")

    metrics_df = pd.DataFrame([result.metrics for result in results])
    importance_df = pd.concat([result.importance for result in results], ignore_index=True)
    importance_matrix = importance_df.pivot_table(
        index="label",
        columns="feature",
        values="importance",
        aggfunc="sum",
        fill_value=0.0,
    )

    metrics_df.to_csv(args.output_dir / "label_metrics.csv", index=False)
    importance_df.to_csv(args.output_dir / "feature_importance_by_label.csv", index=False)
    importance_matrix.to_csv(args.output_dir / "feature_importance_matrix.csv")

    summary = {
        "data_dir": str(args.data_dir),
        "date_range": date_range(merged),
        "feature_rows": int(len(bundle.features)),
        "label_rows": int(len(bundle.labels)),
        "merged_rows": int(len(bundle.merged)),
        "label_columns": label_columns,
        "targets_requested": target_names,
        "targets_completed": metrics_df["label"].tolist(),
        "feature_columns_source": len([column for column in bundle.features.columns if column != "Date"]),
        "feature_columns_used": len(feature_spec.feature_columns),
        "numeric_features": len(numeric_columns),
        "categorical_features": len(categorical_columns),
        "excluded_leakage_columns": feature_spec.leakage_columns,
        "gpu": bool(args.gpu),
        "xgboost_device_requested": "cuda" if args.gpu else "cpu",
        "xgboost_cuda_build": bool(xgb.build_info().get("USE_CUDA")),
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    write_multi_label_visualizations(args.output_dir, metrics_df, importance_df, args.top_n)
    write_multi_markdown(args.output_dir, metrics_df, importance_df, target_names, args)
    print_multi_summary(
        bundle=bundle,
        feature_count_source=summary["feature_columns_source"],
        numeric_count=len(numeric_columns),
        categorical_count=len(categorical_columns),
        label_columns=label_columns,
        target_names=target_names,
        metrics_df=metrics_df,
        gpu=args.gpu,
    )

    print(f"[DONE] Report written to {args.output_dir}")
    print(f"[DONE] Open HTML summary: {args.output_dir / 'report.html'}")
