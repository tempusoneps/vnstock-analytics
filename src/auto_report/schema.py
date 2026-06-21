"""Small data containers used across the report pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    raw: pd.DataFrame
    features: pd.DataFrame
    labels: pd.DataFrame
    merged: pd.DataFrame
    merge_summary: dict[str, Any]


@dataclass
class FeatureSpec:
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    dropped_columns: list[str]
    leakage_columns: list[str]


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    class_names: list[str]
    class_to_index: dict[str, int]
