"""Feature typing, leakage filtering, target encoding, and time splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import DEFAULT_CLASS_ORDER, DEFAULT_LEAKAGE_COLUMNS, MISSING_MARKERS
from .schema import FeatureSpec, SplitData
from .utils import fail


def coerce_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    features = df[feature_columns].copy()
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in feature_columns:
        series = features[column]
        if pd.api.types.is_bool_dtype(series):
            values = series.astype("string")
            features[column] = values.astype("object").where(values.notna(), np.nan)
            categorical_columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(series):
            features[column] = pd.to_numeric(series, errors="coerce")
            numeric_columns.append(column)
            continue

        values = series.astype("string").str.strip()
        values = values.mask(values.str.lower().isin(MISSING_MARKERS))
        converted = pd.to_numeric(values, errors="coerce")
        non_missing_count = int(values.notna().sum())
        numeric_ratio = 0.0 if non_missing_count == 0 else float(converted.notna().sum() / non_missing_count)

        if numeric_ratio >= 0.95:
            features[column] = converted
            numeric_columns.append(column)
        else:
            features[column] = values.astype("object").where(values.notna(), np.nan)
            categorical_columns.append(column)

    for column in numeric_columns:
        features[column] = features[column].replace([np.inf, -np.inf], np.nan)

    return features, numeric_columns, categorical_columns


def build_feature_spec(
    merged: pd.DataFrame,
    label_columns: list[str],
    target: str,
    include_leakage: bool,
) -> FeatureSpec:
    excluded_columns = set(label_columns)
    excluded_columns.add(target)
    excluded_columns.add("Date")

    feature_columns = [
        column
        for column in merged.columns
        if column not in excluded_columns and not column.endswith("_label")
    ]

    leakage_columns = [column for column in DEFAULT_LEAKAGE_COLUMNS if column in feature_columns]
    if not include_leakage:
        feature_columns = [column for column in feature_columns if column not in leakage_columns]

    dropped_columns = []
    for column in list(feature_columns):
        if merged[column].isna().all():
            feature_columns.remove(column)
            dropped_columns.append(column)

    _, numeric_columns, categorical_columns = coerce_feature_columns(merged, feature_columns)
    return FeatureSpec(
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        dropped_columns=dropped_columns,
        leakage_columns=[] if include_leakage else leakage_columns,
    )


def chronological_split(df: pd.DataFrame, target: str, val_size: float, test_size: float) -> SplitData:
    if not 0 < val_size < 0.5:
        fail("--val-size must be between 0 and 0.5")
    if not 0 < test_size < 0.5:
        fail("--test-size must be between 0 and 0.5")
    if val_size + test_size >= 0.8:
        fail("--val-size + --test-size must be below 0.8")

    observed_classes = [value for value in DEFAULT_CLASS_ORDER if value in set(df[target])]
    observed_classes.extend(sorted(value for value in df[target].dropna().unique() if value not in observed_classes))
    class_names = observed_classes
    if len(class_names) < 2:
        fail(f"Target {target!r} has fewer than two classes after cleaning.")

    n_rows = len(df)
    test_count = max(1, int(round(n_rows * test_size)))
    val_count = max(1, int(round(n_rows * val_size)))
    train_count = n_rows - val_count - test_count
    if train_count < max(20, len(class_names) * 3):
        fail(
            "Not enough rows for chronological train/validation/test split. "
            f"rows={n_rows}, train={train_count}, val={val_count}, test={test_count}"
        )

    train_idx = np.arange(0, train_count)
    val_idx = np.arange(train_count, train_count + val_count)
    test_idx = np.arange(train_count + val_count, n_rows)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    return SplitData(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        class_names=class_names,
        class_to_index=class_to_index,
    )


def encoded_target(df: pd.DataFrame, target: str, class_to_index: dict[str, int]) -> np.ndarray:
    encoded = df[target].map(class_to_index)
    if encoded.isna().any():
        unknown = sorted(df.loc[encoded.isna(), target].dropna().unique())
        fail(f"Unknown target classes found: {unknown}")
    return encoded.astype(int).to_numpy()
