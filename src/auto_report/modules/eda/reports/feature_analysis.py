"""EDA feature analysis: distribution shape, outliers, and zero-variance detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ....constants import KEY_COLUMNS


def build_feature_analysis(features: pd.DataFrame, outlier_iqr_factor: float = 1.5) -> pd.DataFrame:
    feature_columns = [col for col in features.columns if col not in KEY_COLUMNS]
    total_rows = len(features)
    rows: list[dict[str, Any]] = []

    for col in feature_columns:
        series = features[col]
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        row: dict[str, Any] = {
            "column": col,
            "dtype": str(series.dtype),
            "rows": total_rows,
            "missing_count": missing_count,
            "missing_rate": float(missing_count / total_rows) if total_rows else 0.0,
            "unique_count": unique_count,
            "is_constant": unique_count <= 1,
            "skewness": pd.NA,
            "kurtosis": pd.NA,
            "outlier_count": pd.NA,
            "outlier_rate": pd.NA,
        }

        if pd.api.types.is_numeric_dtype(series) and not row["is_constant"]:
            numeric = (
                pd.to_numeric(series, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .astype(float)
            )
            if len(numeric) >= 4:
                row["skewness"] = float(stats.skew(numeric))
                row["kurtosis"] = float(stats.kurtosis(numeric))

                q1 = float(numeric.quantile(0.25))
                q3 = float(numeric.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - outlier_iqr_factor * iqr
                    upper = q3 + outlier_iqr_factor * iqr
                    n_outliers = int(((numeric < lower) | (numeric > upper)).sum())
                    row["outlier_count"] = n_outliers
                    row["outlier_rate"] = float(n_outliers / len(numeric))
                else:
                    row["outlier_count"] = 0
                    row["outlier_rate"] = 0.0

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("missing_rate", ascending=False).reset_index(drop=True)
