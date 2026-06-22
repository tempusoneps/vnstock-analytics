"""Base pipeline contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ...config import AutoReportConfig
from ...schema import DatasetBundle


@dataclass
class PipelineContext:
    config: AutoReportConfig
    bundle: DatasetBundle
    output_dir: Path


@dataclass
class PipelineResult:
    name: str
    output_files: list[Path] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class Pipeline(Protocol):
    name: str

    def run(self, context: PipelineContext) -> PipelineResult:
        """Run a report pipeline and write its outputs."""
