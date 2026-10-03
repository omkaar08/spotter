"""
Spotter ML Assessment - Prediction Pipeline

Generates official predictions for:
1. validation_predictions.csv (12,000 loads)
2. data/december_chart_inputs.csv (31 fixed December loads)

Runs official score.py verification.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import FreightFeaturePipeline
from src.train import SpotterEnsemble


def main():
    print("=== RUNNING SPOTTER PREDICTION PIPELINE ===")
    
    # 1. Load pipeline and model artifacts
    pipeline_path = Path('models/pipeline.pkl')
    model_path = Path('models/ensemble_model.pkl')
    
    if not pipeline_path.exists() or not model_path.exists():
        raise FileNotFoundError("Model artifacts missing. Run 'python src/train.py' first.")
        
    with open(pipeline_path, 'rb') as f:
        pipeline: FreightFeaturePipeline = pickle.load(f)
        
    with open(model_path, 'rb') as f:
        model: SpotterEnsemble = pickle.load(f)

    # 2. Read Validation Dataset (12,000 loads)
    val_raw_path = Path('validation.csv')
    if not val_raw_path.exists():
        raise FileNotFoundError("validation.csv not found.")
        
    val_df = pd.read_csv(val_raw_path)
    print(f"Loaded validation dataset: {len(val_df):,} rows")

    # Update daily lookups with validation set market_index & quote_signal
    pipeline.update_daily_lookups(val_df)

    # Transform features and generate predictions
    val_transformed = pipeline.transform(val_df, is_december_input=False)
    val_preds = model.predict(val_transformed[pipeline.feature_columns])

    # Enforce positive rates and build final dataframe
    val_preds_clean = np.maximum(np.round(val_preds, 2), 10.0)

    # Template matching
    template_path = Path('validation-predictions-template (1).csv')
    if template_path.exists():
        sub_df = pd.read_csv(template_path)
        sub_df['predicted_rate'] = val_preds_clean
    else:
        sub_df = pd.DataFrame({
            'load_id': val_df['load_id'],
            'predicted_rate': val_preds_clean
        })

    # Save validation_predictions.csv
    out_val_path = Path('validation_predictions.csv')
    sub_df.to_csv(out_val_path, index=False)
    print(f"Saved validation predictions to {out_val_path} ({len(sub_df):,} rows)")

    # 3. Read December Chart Inputs (31 rows)
    dec_raw_path = Path('december-chart-inputs (1).csv')
    if not dec_raw_path.exists():
        dec_raw_path = Path('data/december_chart_inputs.csv')
        
    dec_df = pd.read_csv(dec_raw_path)
    print(f"Loaded December chart inputs: {len(dec_df)} rows")

    # Transform features for December chart inputs
    dec_transformed = pipeline.transform(dec_df, is_december_input=True)
    dec_preds = model.predict(dec_transformed[pipeline.feature_columns])
    dec_preds_clean = np.maximum(np.round(dec_preds, 2), 10.0)

    dec_df['predicted_rate'] = dec_preds_clean

    # Ensure required columns order
    dec_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    dec_final = dec_df[dec_cols]

    os.makedirs('data', exist_ok=True)
    dec_out_path = Path('data/december_chart_inputs.csv')
    dec_final.to_csv(dec_out_path, index=False)
    print(f"Saved December predictions to {dec_out_path}")

    # Also save a copy at root if needed
    dec_final.to_csv('december_chart_inputs.csv', index=False)

    # 4. Verify with official score.py
    print("\n--- RUNNING OFFICIAL SCORER VALIDATION ---")
    scorer_cmd = f"python score.py --predictions {out_val_path} --december-predictions {dec_out_path}"
    print(f"Executing: {scorer_cmd}")
    os.system(scorer_cmd)


if __name__ == '__main__':
    main()
