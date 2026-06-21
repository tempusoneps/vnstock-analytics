"""CLI entrypoint for the multi-label XGBoost report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .multilabel import run_multi_label


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: true, false, yes, no, 1, 0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train XGBoost against each label column, score predictability, "
            "and visualize feature importance by label."
        )
    )
    parser.add_argument("--data-dir", default="notebooks/VN30F1M", type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("notebooks/VN30F1M/predictability_reports"),
        type=Path,
    )
    parser.add_argument("--raw-file", default="VN30F1M_5m.csv")
    parser.add_argument("--features-file", default="VN30F1M_5m_features.csv")
    parser.add_argument("--labels-file", default="VN30F1M_5m_labels.csv")
    parser.add_argument(
        "--targets",
        default="all",
        help="Comma-separated label columns to predict, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Backward-compatible alias for a single --targets value.",
    )
    parser.add_argument("--test-size", default=0.15, type=float)
    parser.add_argument("--val-size", default=0.15, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument(
        "--max-rows",
        default=0,
        type=int,
        help="Use only the first N usable rows per target after chronological sorting. 0 means all.",
    )
    parser.add_argument(
        "--min-rows-per-target",
        default=500,
        type=int,
        help="Skip a label if it has fewer usable rows than this.",
    )
    parser.add_argument(
        "--include-leakage",
        action="store_true",
        help="Keep known future-looking daily features in model inputs.",
    )
    parser.add_argument("--xgb-n-estimators", default=200, type=int)
    parser.add_argument("--xgb-max-depth", default=4, type=int)
    parser.add_argument("--xgb-learning-rate", default=0.05, type=float)
    parser.add_argument(
        "--gpu",
        default=False,
        type=parse_bool,
        help="Use GPU for XGBoost when available. Accepts true/false. Default: false.",
    )
    parser.add_argument(
        "--top-n",
        default=30,
        type=int,
        help="Number of top features/rows shown in plots and reports.",
    )

    # Legacy options accepted so older commands do not fail after the workflow
    # switched from single-target model comparison to multi-label XGBoost.
    parser.add_argument("--models", default="xgb", help=argparse.SUPPRESS)
    parser.add_argument("--skip-importance", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--importance-sample-size", default=5000, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--permutation-repeats", default=5, type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_multi_label(args)
    except KeyboardInterrupt:
        print("\n[ABORTED] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
