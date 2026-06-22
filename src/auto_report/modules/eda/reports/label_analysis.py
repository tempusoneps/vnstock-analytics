"""Label class distribution and imbalance analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ....constants import KEY_COLUMNS


def build_label_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    label_columns = [col for col in labels.columns if col not in KEY_COLUMNS]
    total_rows = len(labels)
    rows: list[dict[str, Any]] = []

    for col in label_columns:
        series = labels[col].dropna().astype(str)
        counts = series.value_counts()

        if counts.empty:
            rows.append({
                "label": col,
                "class": "(all missing)",
                "count": 0,
                "rate": 0.0,
                "imbalance_ratio": pd.NA,
                "is_imbalanced": True,
            })
            continue

        max_count = int(counts.iloc[0])
        min_count = int(counts.iloc[-1])
        imbalance_ratio = float(max_count / min_count) if min_count > 0 else float("inf")
        is_imbalanced = imbalance_ratio > 3.0

        for class_name, count in counts.items():
            rows.append({
                "label": col,
                "class": str(class_name),
                "count": int(count),
                "rate": float(count / total_rows) if total_rows else 0.0,
                "imbalance_ratio": round(imbalance_ratio, 2),
                "is_imbalanced": is_imbalanced,
            })

    return pd.DataFrame(rows)
