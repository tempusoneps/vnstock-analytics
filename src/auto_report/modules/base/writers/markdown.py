"""Shared Markdown writer helpers."""

from __future__ import annotations

from pathlib import Path

from ....reporting import markdown_table


def write_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
