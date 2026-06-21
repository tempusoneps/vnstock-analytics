"""Model construction, evaluation, plots, and permutation importance."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import fail


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        fail("No usable feature columns after preprocessing.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def selected_models(model_names: list[str], random_state: int, class_count: int) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name in model_names:
        if name == "dummy":
            models[name] = DummyClassifier(strategy="most_frequent")
        elif name == "logreg":
            models[name] = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
            )
        elif name == "xgb":
            try:
                from xgboost import XGBClassifier
            except Exception as exc:  # pragma: no cover - depends on runtime env
                warnings.warn(f"Skipping xgb because import failed: {exc}")
                continue
            models[name] = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=class_count,
                tree_method="hist",
                random_state=random_state,
                n_jobs=0,
            )
        else:
            fail(f"Unsupported model name {name!r}. Use dummy, logreg, xgb.")
    if not models:
        fail("No models selected.")
    return models


def align_proba(model: Pipeline, probabilities: np.ndarray, class_count: int) -> np.ndarray:
    estimator = model.named_steps["model"]
    estimator_classes = getattr(estimator, "classes_", np.arange(probabilities.shape[1]))
    aligned = np.zeros((probabilities.shape[0], class_count), dtype=float)
    for source_index, class_index in enumerate(estimator_classes):
        if int(class_index) < class_count:
            aligned[:, int(class_index)] = probabilities[:, source_index]
    row_sums = aligned.sum(axis=1)
    empty_rows = row_sums == 0
    if np.any(empty_rows):
        aligned[empty_rows, :] = 1.0 / class_count
        row_sums = aligned.sum(axis=1)
    return aligned / row_sums[:, None]


def evaluate_model(
    model_name: str,
    model: Pipeline,
    split_name: str,
    x_split: pd.DataFrame,
    y_true: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_pred = model.predict(x_split)
    class_count = len(class_names)
    metrics = {
        "model": model_name,
        "split": split_name,
        "rows": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": np.nan,
    }

    if hasattr(model, "predict_proba"):
        try:
            proba = align_proba(model, model.predict_proba(x_split), class_count)
            metrics["log_loss"] = log_loss(y_true, proba, labels=list(range(class_count)))
        except Exception as exc:
            warnings.warn(f"Could not calculate log_loss for {model_name}/{split_name}: {exc}")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(class_count)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    report_df.insert(0, "split", split_name)
    report_df.insert(0, "model", model_name)
    return metrics, report_df


def save_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name} test")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=160)
    plt.close(fig)


def calculate_permutation_importance(
    model_name: str,
    model: Pipeline,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    output_dir: Path,
    random_state: int,
    sample_size: int,
    repeats: int,
) -> pd.DataFrame:
    if sample_size > 0 and len(x_val) > sample_size:
        sample_indices = (
            x_val.assign(_target=y_val)
            .groupby("_target", group_keys=False)
            .sample(frac=sample_size / len(x_val), random_state=random_state)
            .index
        )
        if len(sample_indices) == 0:
            sample_indices = x_val.sample(n=sample_size, random_state=random_state).index
        x_sample = x_val.loc[sample_indices]
        y_sample = pd.Series(y_val, index=x_val.index).loc[sample_indices].to_numpy()
    else:
        x_sample = x_val
        y_sample = y_val

    result = permutation_importance(
        model,
        x_sample,
        y_sample,
        scoring="f1_macro",
        n_repeats=repeats,
        random_state=random_state,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": x_val.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "rows_used": len(x_sample),
            "repeats": repeats,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(output_dir / f"permutation_importance_{model_name}.csv", index=False)
    return importance_df
