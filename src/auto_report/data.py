"""Input loading, label alignment, and merge helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import KEY_COLUMNS, MISSING_MARKERS, OHLCV_COLUMNS
from .schema import DatasetBundle
from .utils import fail


def resolve_data_dir(data_dir: Path, raw_file: str, features_file: str, labels_file: str) -> Path:
    required_files = [raw_file, features_file, labels_file]
    if all((data_dir / filename).exists() for filename in required_files):
        return data_dir

    fallback = Path.cwd() / "datasets"
    if data_dir != fallback and all((fallback / filename).exists() for filename in required_files):
        missing = [filename for filename in required_files if not (data_dir / filename).exists()]
        print(
            "[WARN] Requested data-dir is missing files "
            f"{missing}; using fallback data-dir {fallback}"
        )
        return fallback.resolve()

    return data_dir


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing input file: {path}")
    return pd.read_csv(path, low_memory=False)


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if not (
            pd.api.types.is_object_dtype(cleaned[column])
            or pd.api.types.is_string_dtype(cleaned[column])
        ):
            continue
        values = cleaned[column].astype("string").str.strip()
        values = values.mask(values.str.lower().isin(MISSING_MARKERS))
        cleaned[column] = values.astype("object").where(values.notna(), np.nan)
    return cleaned


def validate_label_alignment(raw: pd.DataFrame, labels: pd.DataFrame) -> None:
    if len(raw) != len(labels):
        fail(
            "Raw OHLCV and labels must have the same row count before Date "
            f"assignment. raw={len(raw)}, labels={len(labels)}"
        )

    missing = [column for column in OHLCV_COLUMNS if column not in labels.columns]
    if missing:
        fail(f"Labels file is missing OHLCV columns: {missing}")

    mismatch_counts: dict[str, int] = {}
    for column in OHLCV_COLUMNS:
        left = pd.to_numeric(raw[column], errors="coerce")
        right = pd.to_numeric(labels[column], errors="coerce")
        if column == "Volume":
            mismatches = left.fillna(-1).astype(float) != right.fillna(-1).astype(float)
        else:
            mismatches = ~np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-9,
                atol=1e-9,
            )
        mismatch_counts[column] = int(np.sum(mismatches))

    total_mismatches = sum(mismatch_counts.values())
    if total_mismatches:
        fail(
            "Raw OHLCV and labels are not row-aligned. "
            f"Mismatch counts: {mismatch_counts}"
        )


def prepare_labels(raw: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()

    if "Date" not in labels.columns:
        missing_ohlcv = [column for column in OHLCV_COLUMNS if column not in labels.columns]
        if missing_ohlcv:
            fail("Labels file must contain Date or OHLCV columns.")
        validate_label_alignment(raw, labels)
        labels.insert(0, "Date", raw["Date"].to_numpy())
        return labels

    missing_ohlcv = [column for column in OHLCV_COLUMNS if column not in labels.columns]
    if not missing_ohlcv:
        return labels

    raw_keys = raw[KEY_COLUMNS].copy()
    duplicate_raw_dates = int(raw_keys.duplicated(["Date"]).sum())
    duplicate_label_dates = int(labels.duplicated(["Date"]).sum())
    if duplicate_raw_dates or duplicate_label_dates:
        fail(
            "Cannot attach OHLCV to labels by Date because Date keys are not unique. "
            f"duplicate_raw_dates={duplicate_raw_dates}, duplicate_label_dates={duplicate_label_dates}"
        )

    labels_with_ohlcv = labels.merge(raw_keys, on="Date", how="left", validate="one_to_one")
    missing_rows = int(labels_with_ohlcv[OHLCV_COLUMNS].isna().any(axis=1).sum())
    if missing_rows:
        fail(f"Could not attach OHLCV columns to {missing_rows} label rows by Date.")

    label_columns = [column for column in labels.columns if column != "Date"]
    return labels_with_ohlcv[KEY_COLUMNS + label_columns]


def load_and_merge(data_dir: Path, raw_file: str, features_file: str, labels_file: str) -> DatasetBundle:
    raw_path = data_dir / raw_file
    features_path = data_dir / features_file
    labels_path = data_dir / labels_file

    raw = read_csv(raw_path)
    features = read_csv(features_path)
    labels = read_csv(labels_path)

    for name, frame, required_columns in [
        ("raw", raw, KEY_COLUMNS),
        ("features", features, KEY_COLUMNS),
    ]:
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            fail(f"{name} data is missing required columns: {missing}")

    labels = prepare_labels(raw, labels)

    raw = clean_missing_values(raw)
    features = clean_missing_values(features)
    labels = clean_missing_values(labels)

    duplicate_features = int(features.duplicated(KEY_COLUMNS).sum())
    duplicate_labels = int(labels.duplicated(KEY_COLUMNS).sum())
    if duplicate_features or duplicate_labels:
        fail(
            "Merge keys must be unique. "
            f"duplicate_features={duplicate_features}, duplicate_labels={duplicate_labels}"
        )

    merged = features.merge(
        labels,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_label"),
    )

    missing_labels = int((merged["_merge"] != "both").sum())
    merge_summary = {
        "raw_rows": int(len(raw)),
        "features_rows": int(len(features)),
        "labels_rows": int(len(labels)),
        "merged_rows": int(len(merged)),
        "missing_label_rows": missing_labels,
        "duplicate_feature_keys": duplicate_features,
        "duplicate_label_keys": duplicate_labels,
    }
    if missing_labels:
        fail(f"Merge lost labels for {missing_labels} feature rows.")

    merged = merged.drop(columns=["_merge"])
    return DatasetBundle(raw=raw, features=features, labels=labels, merged=merged, merge_summary=merge_summary)


def normalize_target(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.mask(values.str.lower().isin(MISSING_MARKERS))
    return values
