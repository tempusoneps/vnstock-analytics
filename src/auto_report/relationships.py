"""Feature-label relationship statistics."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer


def relationship_numeric(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_columns: list[str],
    random_state: int,
) -> pd.DataFrame:
    if not numeric_columns:
        return pd.DataFrame()

    x_numeric = x_train[numeric_columns].copy()
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x_numeric)

    mi = mutual_info_classif(
        x_imputed,
        y_train,
        discrete_features=False,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f_scores, f_pvalues = f_classif(x_imputed, y_train)

    rows = []
    classes = sorted(np.unique(y_train))
    for index, column in enumerate(numeric_columns):
        series = pd.to_numeric(x_numeric[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        groups = [series[y_train == klass].dropna().to_numpy() for klass in classes]
        kruskal_stat = np.nan
        kruskal_pvalue = np.nan
        try:
            if sum(len(group) > 0 for group in groups) >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    kruskal = stats.kruskal(*groups, nan_policy="omit")
                kruskal_stat = float(kruskal.statistic)
                kruskal_pvalue = float(kruskal.pvalue)
        except Exception:
            pass

        rows.append(
            {
                "feature": column,
                "mutual_info": float(mi[index]),
                "f_score": float(f_scores[index]) if not math.isnan(float(f_scores[index])) else np.nan,
                "f_pvalue": float(f_pvalues[index]) if not math.isnan(float(f_pvalues[index])) else np.nan,
                "kruskal_stat": kruskal_stat,
                "kruskal_pvalue": kruskal_pvalue,
                "missing_rate": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values(["mutual_info", "f_score"], ascending=False)


def cramers_v(contingency: pd.DataFrame) -> float:
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    n = contingency.to_numpy().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denominator = min((k_corr - 1), (r_corr - 1))
    if denominator <= 0:
        return np.nan
    return float(np.sqrt(phi2_corr / denominator))


def relationship_categorical(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_columns: list[str],
    class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not categorical_columns:
        return pd.DataFrame(), pd.DataFrame()

    y_series = pd.Series([class_names[index] for index in y_train], name="target")
    rows = []
    lift_rows = []
    base_rates = y_series.value_counts(normalize=True).to_dict()

    for column in categorical_columns:
        values = x_train[column].astype("string").fillna("__MISSING__")
        contingency = pd.crosstab(values, y_series)
        if contingency.empty:
            continue

        chi2_value = np.nan
        pvalue = np.nan
        try:
            chi2_result = stats.chi2_contingency(contingency, correction=False)
            chi2_value = float(chi2_result[0])
            pvalue = float(chi2_result[1])
        except Exception:
            pass

        rows.append(
            {
                "feature": column,
                "chi2": chi2_value,
                "pvalue": pvalue,
                "cramers_v": cramers_v(contingency),
                "missing_rate": float(x_train[column].isna().mean()),
                "unique_count": int(values.nunique(dropna=True)),
                "top_value": values.value_counts(dropna=False).index[0],
            }
        )

        if values.nunique(dropna=True) <= 50:
            row_totals = contingency.sum(axis=1)
            for feature_value, counts in contingency.iterrows():
                support = int(row_totals.loc[feature_value])
                if support < 30:
                    continue
                for class_name in class_names:
                    conditional = float(counts.get(class_name, 0) / support)
                    base = base_rates.get(class_name, 0)
                    lift = np.nan if base == 0 else conditional / base
                    lift_rows.append(
                        {
                            "feature": column,
                            "value": feature_value,
                            "target": class_name,
                            "support": support,
                            "conditional_rate": conditional,
                            "base_rate": base,
                            "lift": lift,
                        }
                    )

    relationship_df = pd.DataFrame(rows)
    if not relationship_df.empty:
        relationship_df = relationship_df.sort_values(["cramers_v", "chi2"], ascending=False)

    lift_df = pd.DataFrame(lift_rows)
    if not lift_df.empty:
        lift_df = lift_df.sort_values(["lift", "support"], ascending=[False, False])
    return relationship_df, lift_df
