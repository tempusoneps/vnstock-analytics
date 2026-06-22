"""Label-column statistics report."""

from __future__ import annotations

import pandas as pd

from ....constants import KEY_COLUMNS


def build_label_statistics(labels: pd.DataFrame) -> pd.DataFrame:
    label_columns = [column for column in labels.columns if column not in KEY_COLUMNS]
    rows = []
    total_rows = len(labels)

    for column in label_columns:
        series = labels[column]
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        row = {
            "column": column,
            "dtype": str(series.dtype),
            "rows": int(total_rows),
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "missing_rate": 0.0 if total_rows == 0 else float(missing_count / total_rows),
            "unique_count": unique_count,
            "top_value": "",
            "top_value_count": pd.NA,
            "top_value_rate": pd.NA,
        }

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
