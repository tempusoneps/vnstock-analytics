"""Pipeline for multi-label XGBoost reports."""

from __future__ import annotations

import json

import pandas as pd
import xgboost as xgb

from ...constants import KEY_COLUMNS
from ...features import build_feature_spec, coerce_feature_columns
from ...multilabel import (
    TargetResult,
    parse_target_names,
    print_multi_summary,
    run_one_target,
)
from ...reporting import date_range
from ...utils import fail
from ..base.pipeline import PipelineContext, PipelineResult
from .reports.feature_importance import build_importance_matrix
from .writers.csv import write_xgboost_csv
from .writers.html import write_xgboost_html
from .writers.markdown import write_xgboost_markdown


class XGBoostPipeline:
    name = "xgboost"

    def run(self, context: PipelineContext) -> PipelineResult:
        args = context.config
        bundle = context.bundle
        label_columns = [column for column in bundle.labels.columns if column not in KEY_COLUMNS]
        target_names = parse_target_names(args.targets, args.target, label_columns)

        merged = bundle.merged.copy()
        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
        merged = merged.sort_values("Date").reset_index(drop=True)

        feature_spec = build_feature_spec(
            merged,
            label_columns=label_columns,
            target=target_names[0],
            include_leakage=args.include_leakage,
        )
        x_all, numeric_columns, categorical_columns = coerce_feature_columns(merged, feature_spec.feature_columns)
        feature_spec.numeric_columns = numeric_columns
        feature_spec.categorical_columns = categorical_columns

        print(f"[INFO] Training XGBoost for {len(target_names)} target label(s)")
        results: list[TargetResult] = []
        for target in target_names:
            print(f"[INFO] Target: {target}")
            result = run_one_target(target, merged, x_all, numeric_columns, categorical_columns, args)
            if result is not None:
                results.append(result)

        if not results:
            fail("No targets were completed.")

        metrics_df = pd.DataFrame([result.metrics for result in results])
        importance_df = pd.concat([result.importance for result in results], ignore_index=True)
        importance_matrix = build_importance_matrix(importance_df)

        output_files = []
        output_files.extend(write_xgboost_csv(context.output_dir, metrics_df, importance_df, importance_matrix))
        output_files.extend(write_xgboost_html(context.output_dir, metrics_df, importance_df, args.top_n))
        output_files.append(write_xgboost_markdown(context.output_dir, metrics_df, importance_df, target_names, args))

        summary = {
            "data_dir": str(args.data_dir),
            "date_range": date_range(merged),
            "feature_rows": int(len(bundle.features)),
            "label_rows": int(len(bundle.labels)),
            "merged_rows": int(len(bundle.merged)),
            "label_columns": label_columns,
            "targets_requested": target_names,
            "targets_completed": metrics_df["label"].tolist(),
            "feature_columns_source": len([column for column in bundle.features.columns if column != "Date"]),
            "feature_columns_used": len(feature_spec.feature_columns),
            "numeric_features": len(numeric_columns),
            "categorical_features": len(categorical_columns),
            "excluded_leakage_columns": feature_spec.leakage_columns,
            "gpu": bool(args.gpu),
            "xgboost_device_requested": "cuda" if args.gpu else "cpu",
            "xgboost_cuda_build": bool(xgb.build_info().get("USE_CUDA")),
        }
        summary_path = context.output_dir / "xgboost_dataset_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        output_files.append(summary_path)

        print_multi_summary(
            bundle=bundle,
            feature_count_source=summary["feature_columns_source"],
            numeric_count=len(numeric_columns),
            categorical_count=len(categorical_columns),
            label_columns=label_columns,
            target_names=target_names,
            metrics_df=metrics_df,
            gpu=args.gpu,
        )
        return PipelineResult(name=self.name, output_files=output_files, summary=summary)
