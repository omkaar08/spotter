"""
Spotter ML Assessment - Model Evaluation & Residual Analysis

Generates comprehensive evaluation charts, feature importance rankings,
residual diagnostics, and segment error analysis.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import FreightFeaturePipeline
from src.train import SpotterEnsemble


def main():
    print("=== GENERATING EVALUATION DIAGNOSTICS & CHARTS ===")
    os.makedirs('outputs', exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.sans-serif': 'Inter, Arial, sans-serif', 'font.size': 11})

    # Load data and pipeline
    train_raw = pd.read_csv('train-test.csv')
    with open('models/pipeline.pkl', 'rb') as f:
        pipeline: FreightFeaturePipeline = pickle.load(f)
    with open('models/ensemble_model.pkl', 'rb') as f:
        model: SpotterEnsemble = pickle.load(f)

    train_df = pipeline.transform(train_raw)
    feature_cols = pipeline.feature_columns
    y = train_df['posted_rate']

    # Temporal OOT Split (Jan-Aug vs Sept-Oct)
    tr_mask = train_df['month'] <= 8
    va_mask = train_df['month'] > 8

    X_va = train_df.loc[va_mask, feature_cols]
    y_va = y.loc[va_mask]
    
    # Fit temporary OOT model for clean evaluation
    oot_model = SpotterEnsemble().fit(train_df.loc[tr_mask, feature_cols], y.loc[tr_mask])
    preds_va = oot_model.predict(X_va)
    residuals = y_va - preds_va

    # 1. Feature Importance Plot from CatBoost
    cat_model = oot_model.cat
    importances = cat_model.get_feature_importance()
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    feat_imp.to_csv('outputs/feature_importances.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    top_15 = feat_imp.head(15)
    sns.barplot(data=top_15, x='importance', y='feature', palette='viridis', ax=ax)
    ax.set_title('Top 15 Predictive Features (CatBoost Relative Importance)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Relative Importance Score (%)')
    ax.set_ylabel('')
    fig.tight_layout()
    fig.savefig('outputs/feature_importances.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved feature importances plot to outputs/feature_importances.png")

    # 2. Residual Diagnostics Plot (Actual vs Predicted & Residual Distribution)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    
    # Scatter plot
    axes[0].scatter(y_va, preds_va, alpha=0.35, color='#064A56', edgecolors='none', s=15)
    max_val = max(y_va.max(), preds_va.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction (1:1)')
    axes[0].set_title('Actual vs Predicted Freight Rate ($)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Actual Posted Rate ($)')
    axes[0].set_ylabel('Predicted Rate ($)')
    axes[0].legend(frameon=True)

    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color='#064A56', bins=50)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residual Error Distribution (Actual - Predicted)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Residual Error ($)')
    axes[1].set_ylabel('Frequency')

    fig.tight_layout()
    fig.savefig('outputs/residual_analysis.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved residual analysis plot to outputs/residual_analysis.png")

    # 3. Segment Error Breakdown by Equipment and Distance Decile
    val_analysis = train_raw.loc[va_mask].copy()
    val_analysis['actual'] = y_va
    val_analysis['pred'] = preds_va
    val_analysis['abs_error'] = np.abs(val_analysis['actual'] - val_analysis['pred'])

    eq_summary = val_analysis.groupby('equipment')['abs_error'].agg(
        Mean_MAE='mean', Median_AE='median', Count='count'
    ).reset_index()

    val_analysis['distance_bin'] = pd.qcut(val_analysis['distance'], q=5, labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    dist_summary = val_analysis.groupby('distance_bin', observed=False)['abs_error'].agg(
        Mean_MAE='mean', Median_AE='median', Count='count'
    ).reset_index()

    # Save summary report text
    with open('outputs/evaluation_report.txt', 'w') as f:
        f.write("=== SPOTTER ML MODEL EVALUATION REPORT ===\n\n")
        f.write(f"Validation Sample Size: {len(val_analysis):,} loads (Sept-Oct 2025)\n")
        f.write(f"Overall MAE: ${val_analysis['abs_error'].mean():.2f}\n")
        f.write(f"Overall Median AE: ${val_analysis['abs_error'].median():.2f}\n\n")
        f.write("--- Error by Equipment Type ---\n")
        f.write(eq_summary.to_string(index=False) + "\n\n")
        f.write("--- Error by Distance Quintiles ---\n")
        f.write(dist_summary.to_string(index=False) + "\n")

    print("Saved evaluation report to outputs/evaluation_report.txt")


if __name__ == '__main__':
    main()
