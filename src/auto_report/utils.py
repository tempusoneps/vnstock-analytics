"""Generic utilities shared by modules."""

from __future__ import annotations

from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"[ERROR] {message}")


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()
