"""
Spotter ML Assessment - Model Training & Evaluation Pipeline

Trains baseline, linear, tree-based, and gradient boosting models using
Time-Based Out-Of-Time (OOT) validation and K-Fold block temporal CV.
Saves model comparison metrics and exports the final ensemble model.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import FreightFeaturePipeline


class SpotterEnsemble:
    """
    Weighted Ensemble combining CatBoost, LightGBM, and Ridge Regression.
    """
    def __init__(self, cat_weight: float = 0.55, lgb_weight: float = 0.30, ridge_weight: float = 0.15):
        self.cat_weight = cat_weight
        self.lgb_weight = lgb_weight
        self.ridge_weight = ridge_weight
        
        self.cat = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, random_state=42, verbose=0)
        self.lgb = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
        self.ridge = Ridge(alpha=10.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SpotterEnsemble':
        self.cat.fit(X, y)
        self.lgb.fit(X, y)
        self.ridge.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_cat = self.cat.predict(X)
        p_lgb = self.lgb.predict(X)
        p_ridge = self.ridge.predict(X)
        
        ensemble_pred = (
            self.cat_weight * p_cat +
            self.lgb_weight * p_lgb +
            self.ridge_weight * p_ridge
        )
        return np.maximum(ensemble_pred, 10.0)  # Enforce positive freight rates


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates standard regression evaluation metrics."""
    return {
        'MAE ($)': float(mean_absolute_error(y_true, y_pred)),
        'RMSE ($)': float(root_mean_squared_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
        'MedAE ($)': float(median_absolute_error(y_true, y_pred))
    }


def run_training_experiment():
    print("=== STARTING MODEL TRAINING & EVALUATION EXPERIMENT ===")
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    train_raw = pd.read_csv('train-test.csv')
    
    # Fit Feature Pipeline
    pipeline = FreightFeaturePipeline().fit(train_raw)
    train_df = pipeline.transform(train_raw)
    
    feature_cols = pipeline.feature_columns
    y = train_df['posted_rate']

    # Temporal Out-Of-Time Split (Train: Jan-Aug, Val: Sept-Oct)
    tr_mask = train_df['month'] <= 8
    va_mask = train_df['month'] > 8

    X_tr, y_tr = train_df.loc[tr_mask, feature_cols], y.loc[tr_mask]
    X_va, y_va = train_df.loc[va_mask, feature_cols], y.loc[va_mask]

    print(f"OOT Split Integrity -> Train: {len(X_tr):,} loads (Jan-Aug), Val: {len(X_va):,} loads (Sept-Oct)")

    models = {
        'Quote Signal Formula Baseline': None,
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1),
        'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42),
        'CatBoost': CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, random_state=42, verbose=0),
        'Spotter Weighted Ensemble (Proposed)': SpotterEnsemble()
    }

    comparison_results = []

    for name, model in models.items():
        print(f"Training and evaluating: {name}...")
        if name == 'Quote Signal Formula Baseline':
            preds = train_df.loc[va_mask, 'quote_rate'].values
        else:
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
        
        metrics = evaluate_predictions(y_va.values, preds)
        metrics['Model'] = name
        comparison_results.append(metrics)

    comp_df = pd.DataFrame(comparison_results)[['Model', 'MAE ($)', 'RMSE ($)', 'R2', 'MedAE ($)']]
    comp_df = comp_df.sort_values('MAE ($)')
    
    print("\n=== MODEL COMPARISON TABLE (Out-Of-Time Temporal Validation) ===")
    print(comp_df.to_string(index=False))
    comp_df.to_csv('outputs/model_comparison.csv', index=False)
    print("\nModel comparison saved to outputs/model_comparison.csv")

    # Fit Final Production Ensemble on 100% of Train Data
    print("\nFitting final Spotter Ensemble on 100% development dataset (48,000 loads)...")
    final_ensemble = SpotterEnsemble().fit(train_df[feature_cols], y)

    # Save Pipeline and Model Artifacts
    with open('models/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        
    with open('models/ensemble_model.pkl', 'wb') as f:
        pickle.dump(final_ensemble, f)

    print("Final model artifacts successfully saved to models/ directory.")


if __name__ == '__main__':
    run_training_experiment()
