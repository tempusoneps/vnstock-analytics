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
ANALYTICS_DATASET = 'VN30F1M_5m_ready.csv'
CURRENT_DIR = os.getcwd()
    

def load_analytics_dataset():
    csv_feature_file = str(CURRENT_DIR) + '/' + FEATURE_DATASET
    csv_final_file = str(CURRENT_DIR) + '/' + ANALYTICS_DATASET
    is_file = os.path.isfile(csv_final_file)
    if is_file:
        dataset = pd.read_csv(csv_final_file, index_col='Date', parse_dates=True)
    else:
        if os.path.isfile(csv_feature_file):
            df = pd.read_csv(csv_feature_file, index_col='Date', parse_dates=True)
            dataset = do_label_data(df)
            dataset.to_csv(csv_final_file)
        else:
            dataset = None
    return dataset

def do_label_data(df):
    with urlopen(RULE_URL) as response:
        rules = json.loads(response.read().decode('utf-8'))
    rule_id = "no-overnight-sl033-tp132-tsl035-fc1425"
    rule = next((r for r in rules["rules"] if r["id"] == rule_id), None)
    if not rule:
        return None
    label_data = df.copy()
    new_entry_allowed = []
    for i, row in label_data.iterrows():
        current_date = row.name.strftime('%Y-%m-%d ').format()
        current_time = row.name
        data_to_end_day = label_data[(label_data.index > current_time) & (label_data.index < current_date + ' 14:30:00')]
        if not len(data_to_end_day):
            new_entry_allowed.append("")
            continue
        #
        entry_price = row['Close']
        long_sl = entry_price - entry_price * rule['risk_management']['stop_loss']['value'] / 100
        short_sl = entry_price + entry_price * rule['risk_management']['stop_loss']['value'] / 100
        longable = shortable = True
        if data_to_end_day['High'].max() >= short_sl:
            shortable = False
        if data_to_end_day['Low'].min() <= long_sl:
            longable = False
        #
        if longable and shortable:
            new_entry_allowed.append('Sideway')
        elif longable:
            new_entry_allowed.append('Bullish')
        elif shortable:
            new_entry_allowed.append('Bearish')
        else:
            new_entry_allowed.append("None")
    #
    label_data['allow_entry'] = new_entry_allowed
    return label_data

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
    axes[0].set_xlabel('Color')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Label')
    
    # 2. 100% Stacked Bar Chart
    ct_norm.plot(kind='bar', stacked=True, ax=axes[1])
    axes[1].set_title('Percentage Distribution')
    axes[1].set_xlabel('Color')
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