"""Label predictability report helpers for XGBoost."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LabelPredictabilityReport:
    metrics: pd.DataFrame
    importance: pd.DataFrame
