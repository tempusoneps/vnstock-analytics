"""CLI entrypoint for auto report modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, ReportConfigFile
from .data import load_and_merge, resolve_data_dir
from .modules.base.pipeline import PipelineContext
from .modules.eda.pipeline import EdaPipeline
from .modules.statistics.pipeline import StatisticsPipeline
from .modules.xgboost.pipeline import XGBoostPipeline
from .utils import normalize_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dataset statistics and XGBoost label predictability reports."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        type=Path,
        help=f"Path to JSON config file. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config_file = ReportConfigFile.load(args.config)
        pipeline_by_module = {
            "eda": EdaPipeline(),
            "statistics": StatisticsPipeline(),
            "xgboost": XGBoostPipeline(),
        }

        output_dirs = set()
        for module_name in config_file.module_names():
            config = config_file.module_config(module_name)
            config.data_dir = normalize_path(config.data_dir)
            config.data_dir = resolve_data_dir(
                config.data_dir,
                config.raw_file,
                config.features_file,
                config.labels_file,
            )
            config.output_dir = normalize_path(config.output_dir)
            config.output_dir.mkdir(parents=True, exist_ok=True)
            output_dirs.add(config.output_dir)

            print(f"[INFO] Loading data for module {module_name} from {config.data_dir}")
            bundle = load_and_merge(config.data_dir, config.raw_file, config.features_file, config.labels_file)
            context = PipelineContext(config=config, bundle=bundle, output_dir=config.output_dir)
            pipeline = pipeline_by_module[module_name]
            print(f"[INFO] Running module: {pipeline.name}")
            pipeline.run(context)

        for output_dir in sorted(output_dirs):
            print(f"[DONE] Report written to {output_dir}")
            if (output_dir / "eda_report.html").exists():
                print(f"[DONE] Open EDA summary: {output_dir / 'eda_report.html'}")
            if (output_dir / "statistics_report.html").exists():
                print(f"[DONE] Open statistics summary: {output_dir / 'statistics_report.html'}")
            if (output_dir / "xgboost_report.html").exists():
                print(f"[DONE] Open XGBoost summary: {output_dir / 'xgboost_report.html'}")
    except KeyboardInterrupt:
        print("\n[ABORTED] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
