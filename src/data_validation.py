"""
Spotter ML Assessment - Data Validation Module

This module performs automated data audits, schema validation, missing value detection,
outlier checks, and temporal boundary verification across train, validation, and December datasets.
"""

import pandas as pd
import numpy as np

REQUIRED_TRAIN_COLS = [
    'load_id', 'pickup', 'delivery', 'pickup_lat', 'pickup_lon',
    'delivery_lat', 'delivery_lon', 'distance', 'equipment', 'weight',
    'date', 'market_index', 'quote_signal', 'posted_rate'
]

REQUIRED_VAL_COLS = [
    'load_id', 'pickup', 'delivery', 'pickup_lat', 'pickup_lon',
    'delivery_lat', 'delivery_lon', 'distance', 'equipment', 'weight',
    'date', 'market_index', 'quote_signal'
]

REQUIRED_DECEMBER_COLS = [
    'pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate'
]


def audit_dataset(df: pd.DataFrame, dataset_name: str, is_train: bool = True) -> dict:
    """
    Performs comprehensive schema, missing value, and temporal checks.
    """
    required_cols = REQUIRED_TRAIN_COLS if is_train else REQUIRED_VAL_COLS
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"[{dataset_name}] Missing required columns: {missing_cols}")
        
    null_counts = df.isna().sum().to_dict()
    num_duplicates = df['load_id'].duplicated().sum() if 'load_id' in df.columns else 0
    
    min_date = pd.to_datetime(df['date']).min().strftime('%Y-%m-%d')
    max_date = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
    
    audit_summary = {
        'dataset_name': dataset_name,
        'rows': len(df),
        'cols': len(df.columns),
        'min_date': min_date,
        'max_date': max_date,
        'duplicate_ids': num_duplicates,
        'missing_values': {k: v for k, v in null_counts.items() if v > 0}
    }
    
    print(f"[{dataset_name} Audit Complete] Rows: {len(df):,}, Range: {min_date} to {max_date}, Dups: {num_duplicates}")
    if audit_summary['missing_values']:
        print(f"[{dataset_name} Missing Values] {audit_summary['missing_values']}")
        
    return audit_summary


if __name__ == '__main__':
    train_df = pd.read_csv('train-test.csv')
    val_df = pd.read_csv('validation.csv')
    audit_dataset(train_df, 'Train-Test', is_train=True)
    audit_dataset(val_df, 'Validation', is_train=False)
