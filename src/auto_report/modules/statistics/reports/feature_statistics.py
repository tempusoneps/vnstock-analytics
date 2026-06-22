"""Feature-column statistics report."""

from __future__ import annotations

import pandas as pd

from ....reporting import describe_feature_columns


def build_feature_statistics(features: pd.DataFrame) -> pd.DataFrame:
    return describe_feature_columns(features)
