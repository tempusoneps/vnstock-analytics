"""Feature importance report helpers for XGBoost."""

from __future__ import annotations

import pandas as pd


def build_importance_matrix(importance_df: pd.DataFrame) -> pd.DataFrame:
    return importance_df.pivot_table(
        index="label",
        columns="feature",
        values="importance",
        aggfunc="sum",
        fill_value=0.0,
    )
