import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from urllib.request import urlopen
import json
import seaborn as sns
import matplotlib.pyplot as plt


RULE_URL = 'https://raw.githubusercontent.com/tempusoneps/trading-rules/refs/heads/main/VN30F1M/close_position_rules.json'
OHCLV_DATASET = 'VN30F1M_5m.csv'
FEATURE_DATASET = 'VN30F1M_5m_features.csv'
LABEL_DATASET = 'VN30F1M_5m_labels.csv'
ANALYTICS_DATASET = 'VN30F1M_5m_ready.csv'
CURRENT_DIR = os.getcwd()
KEY_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
OHLCV_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

# constraints
BUY_MEET_SL = 'SL(Buy)'
SELL_MEET_SL   = 'SL(Sell)'
    

def load_analytics_dataset():
    data_dir = resolve_dataset_dir()
    csv_feature_file = data_dir / FEATURE_DATASET
    csv_label_file = data_dir / LABEL_DATASET
    csv_final_file = data_dir / ANALYTICS_DATASET
    is_file = os.path.isfile(csv_final_file)
    if is_file:
        dataset = pd.read_csv(csv_final_file, index_col='Date', parse_dates=True)
    else:
        if os.path.isfile(csv_feature_file) and os.path.isfile(csv_label_file):
            features = read_dataset_csv(csv_feature_file)
            labels = read_dataset_csv(csv_label_file)
            dataset = merge_feature_label_dataset(features, labels)
            dataset.to_csv(csv_final_file)
        else:
            dataset = None
    return dataset

def resolve_dataset_dir():
    current_dir = Path(CURRENT_DIR).resolve()
    candidates = []
    for path in [current_dir] + list(current_dir.parents):
        candidates.append(path / 'datasets')
    candidates.append(current_dir)

    for data_dir in candidates:
        if (data_dir / FEATURE_DATASET).is_file() and (data_dir / LABEL_DATASET).is_file():
            return data_dir

    return current_dir

def read_dataset_csv(path):
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    parse_dates = ['Date'] if 'Date' in columns else None
    return pd.read_csv(path, parse_dates=parse_dates)

def merge_feature_label_dataset(features, labels):
    missing_feature_columns = [column for column in KEY_COLUMNS if column not in features.columns]
    if missing_feature_columns:
        raise ValueError(f'Feature dataset is missing required columns: {missing_feature_columns}')

    labels = prepare_label_dataset(features, labels)
    merge_keys = KEY_COLUMNS if all(column in labels.columns for column in KEY_COLUMNS) else ['Date']

    duplicate_features = features.duplicated(merge_keys).sum()
    duplicate_labels = labels.duplicated(merge_keys).sum()
    if duplicate_features or duplicate_labels:
        raise ValueError(
            'Merge keys must be unique. '
            f'duplicate_features={duplicate_features}, duplicate_labels={duplicate_labels}'
        )

    dataset = features.merge(
        labels,
        on=merge_keys,
        how='left',
        validate='one_to_one',
        indicator=True,
        suffixes=('', '_label'),
    )
    missing_labels = (dataset['_merge'] != 'both').sum()
    if missing_labels:
        raise ValueError(f'Merge lost labels for {missing_labels} feature rows.')

    dataset = dataset.drop(columns=['_merge'])
    return dataset.set_index('Date').sort_index()

def prepare_label_dataset(features, labels):
    labels = labels.copy()
    if 'Date' in labels.columns:
        labels['Date'] = pd.to_datetime(labels['Date'], errors='coerce')
        return labels

    if all(column in labels.columns for column in OHLCV_COLUMNS):
        aligned_labels = align_labels_to_features_by_ohlcv(features, labels)
        if aligned_labels is not None:
            return aligned_labels

    if len(features) != len(labels):
        raise ValueError(
            'Label dataset must contain Date, align to feature OHLCV rows, '
            'or have the same row count as the feature dataset. '
            f'features={len(features)}, labels={len(labels)}'
        )

    if all(column in labels.columns for column in OHLCV_COLUMNS):
        mismatch_counts = {}
        for column in OHLCV_COLUMNS:
            left = pd.to_numeric(features[column], errors='coerce')
            right = pd.to_numeric(labels[column], errors='coerce')
            if column == 'Volume':
                mismatches = left.fillna(-1).astype(float) != right.fillna(-1).astype(float)
            else:
                mismatches = ~np.isclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    equal_nan=True,
                    rtol=1e-9,
                    atol=1e-9,
                )
            mismatch_counts[column] = int(np.sum(mismatches))
        if sum(mismatch_counts.values()):
            raise ValueError(
                'Feature and label OHLCV rows are not aligned. '
                f'Mismatch counts: {mismatch_counts}'
            )

    labels.insert(0, 'Date', pd.to_datetime(features['Date'], errors='coerce').to_numpy())
    return labels

def align_labels_to_features_by_ohlcv(features, labels):
    feature_keys = features[OHLCV_COLUMNS].reset_index(drop=True)
    label_keys = labels[OHLCV_COLUMNS].reset_index(drop=True)

    label_rows_by_key = {}
    for label_index, row in label_keys.iterrows():
        key = ohlcv_key(row)
        label_rows_by_key.setdefault(key, []).append(label_index)

    matched_label_indices = []
    last_label_index = -1
    for _, row in feature_keys.iterrows():
        key = ohlcv_key(row)
        candidate_indices = label_rows_by_key.get(key, [])
        next_label_index = None
        while candidate_indices:
            candidate_index = candidate_indices.pop(0)
            if candidate_index > last_label_index:
                next_label_index = candidate_index
                break

        if next_label_index is None:
            return None

        matched_label_indices.append(next_label_index)
        last_label_index = next_label_index

    aligned_labels = labels.iloc[matched_label_indices].reset_index(drop=True).copy()
    if not ohlcv_frames_match(feature_keys, aligned_labels[OHLCV_COLUMNS].reset_index(drop=True)):
        return None

    aligned_labels.insert(0, 'Date', pd.to_datetime(features['Date'], errors='coerce').to_numpy())
    return aligned_labels

def ohlcv_key(row):
    values = []
    for column in OHLCV_COLUMNS:
        value = row[column]
        values.append(None if pd.isna(value) else float(value))
    return tuple(values)

def ohlcv_frames_match(left_frame, right_frame):
    for column in OHLCV_COLUMNS:
        left = pd.to_numeric(left_frame[column], errors='coerce')
        right = pd.to_numeric(right_frame[column], errors='coerce')
        if column == 'Volume':
            mismatches = left.fillna(-1).astype(float) != right.fillna(-1).astype(float)
        else:
            mismatches = ~np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-9,
                atol=1e-9,
            )
        if np.any(mismatches):
            return False
    return True

def show_3_distribution_charts(ts1, ts2):
    if len(ts1.unique()) > 10:
        print("The first column has more than 10 unique value. You should use line or KDE plot!")
        return
    ct = pd.crosstab(ts1, ts2)
    # Normalize (percentage)
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    
    # ====== Plot ======
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Grouped Bar Chart (Count)
    ct.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Count Distribution')
    axes[0].set_xlabel(ts1.name)
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Label')
    
    # 2. 100% Stacked Bar Chart
    ct_norm.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('Percentage Distribution')
    axes[1].set_xlabel(ts1.name)
    axes[1].set_ylabel('Ratio')
    axes[1].legend(title='Label')
    
    # 3. Heatmap (manual using imshow)
    im = axes[2].imshow(ct.values)
    
    # ticks
    axes[2].set_xticks(np.arange(len(ct.columns)))
    axes[2].set_yticks(np.arange(len(ct.index)))
    
    axes[2].set_xticklabels(ct.columns)
    axes[2].set_yticklabels(ct.index)
    
    axes[2].set_title('Heatmap Pattern')
    
    # annotate values
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            axes[2].text(j, i, ct.values[i, j],
                         ha="center", va="center")
    
    plt.tight_layout()
    plt.show()

def show_3_sns_charts(df, col_1, col_2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title = f'{col_1.upper()} & {col_2.upper()} Distribution'

    # 1. KDE plot
    sns.kdeplot(data=df, x=col_1, hue=col_2, ax=axes[0])
    axes[0].set_title(f'{title} (KDE)')
    
    # 2. Box plot
    sns.boxplot(data=df, x=col_2, y=col_1, ax=axes[1])
    axes[1].set_title(f'{title} (Boxplot)')
    
    # 3. Violin plot
    sns.violinplot(data=df, x=col_2, y=col_1, ax=axes[2])
    axes[2].set_title(f'{title} (Violin)')
    
    plt.tight_layout()
    plt.show()
