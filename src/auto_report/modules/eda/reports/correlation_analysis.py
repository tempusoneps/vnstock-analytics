"""Feature-feature Pearson correlation matrix and high-correlation pair detection."""

from __future__ import annotations

import pandas as pd

from ....constants import KEY_COLUMNS


def build_correlation_matrix(features: pd.DataFrame, max_features: int = 50) -> pd.DataFrame:
    numeric_cols = [
        col for col in features.columns
        if col not in KEY_COLUMNS and pd.api.types.is_numeric_dtype(features[col])
    ]
    if not numeric_cols:
        return pd.DataFrame()
    numeric_cols = numeric_cols[:max_features]
    return features[numeric_cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")


def build_top_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    if corr_matrix.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])

    cols = corr_matrix.columns.tolist()
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                rows.append({"feature_a": cols[i], "feature_b": cols[j], "correlation": float(val)})

    if not rows:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])

    df = pd.DataFrame(rows)
    return df.iloc[df["correlation"].abs().argsort()[::-1].values].reset_index(drop=True)
