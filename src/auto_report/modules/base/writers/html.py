"""Shared HTML writer helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def dataframe_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="table", float_format=lambda value: f"{value:.6g}")


def write_html(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
