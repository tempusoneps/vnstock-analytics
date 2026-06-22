"""Pipeline for dataset statistics reports."""

from __future__ import annotations

from .reports.feature_statistics import build_feature_statistics
from .reports.label_statistics import build_label_statistics
from .writers.csv import write_statistics_csv
from .writers.html import write_statistics_html
from .writers.markdown import write_statistics_markdown
from ..base.pipeline import PipelineContext, PipelineResult


class StatisticsPipeline:
    name = "statistics"

    def run(self, context: PipelineContext) -> PipelineResult:
        feature_stats = build_feature_statistics(context.bundle.features)
        label_stats = build_label_statistics(context.bundle.labels)

        output_files = []
        output_files.extend(write_statistics_csv(context.output_dir, feature_stats, label_stats))
        output_files.append(
            write_statistics_markdown(
                context.output_dir,
                feature_stats,
                label_stats,
                context.config.top_n,
            )
        )
        output_files.append(
            write_statistics_html(
                context.output_dir,
                feature_stats,
                label_stats,
                context.config.top_n,
            )
        )

        summary = {
            "feature_columns": int(len(feature_stats)),
            "label_columns": int(len(label_stats)),
            "feature_columns_with_missing": int((feature_stats["missing_count"] > 0).sum())
            if not feature_stats.empty
            else 0,
            "label_columns_with_missing": int((label_stats["missing_count"] > 0).sum())
            if not label_stats.empty
            else 0,
        }
        return PipelineResult(name=self.name, output_files=output_files, summary=summary)
