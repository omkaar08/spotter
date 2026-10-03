# Freight Rate Prediction Solution

This repository contains the code, data pipeline, and predictions for the Spotter AI Freight Rate Prediction ML challenge.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models & Run Out-Of-Time Validation
```bash
python src/train.py
```
Trains Ridge, Random Forest, LightGBM, XGBoost, CatBoost, and the weighted ensemble on a temporal Out-Of-Time (OOT) split (Jan-Aug 2025 Train -> Sept-Oct 2025 Validation). Saves model binaries to `models/` and metrics to `outputs/model_comparison.csv`.

### 3. Generate Predictions & Run Scorer
```bash
python src/predict.py
```
Generates rate predictions for `validation_predictions.csv` (12,000 loads) and `data/december_chart_inputs.csv` (31 loads), then runs `score.py` for verification.

### 4. Diagnostic Plots & PDF Report
```bash
python src/evaluate.py
python generate_pdf_report.py
```
Generates feature importance plots (`outputs/feature_importances.png`), residual plots (`outputs/residual_analysis.png`), and compiles `Spotter_ML_Assessment_Report.pdf`.

---

## Validation Benchmark Results (Sept–Oct 2025 Holdout)

| Model | MAE ($) | RMSE ($) | R² | MedAE ($) |
|---|---|---|---|---|
| **CatBoost** | **$147.50** | **$619.90** | **0.8350** | **$60.57** |
| **Spotter Weighted Ensemble (Final)** | **$149.56** | **$625.04** | **0.8322** | **$64.07** |
| LightGBM | $157.61 | $646.88 | 0.8203 | $60.62 |
| XGBoost | $180.43 | $690.25 | 0.7954 | $72.78 |
| Random Forest | $186.15 | $678.37 | 0.8024 | $60.30 |
| Ridge Regression | $200.96 | $639.81 | 0.8242 | $127.83 |
| Quote Signal Formula Baseline | $246.02 | $711.80 | 0.7824 | $60.44 |

---

## Repository Structure

```
├── train-test.csv                       # Development dataset (48,000 loads)
├── validation.csv                       # Validation dataset (12,000 loads)
├── score.py                             # Official scoring script
├── requirements.txt                     # Package requirements
├── generate_pdf_report.py               # PDF report script
├── Spotter_ML_Assessment_Report.pdf     # Technical PDF report
├── approach.md                          # Methodology documentation
│
├── src/
│   ├── data_validation.py               # Data schema and missing value check
│   ├── preprocessing.py                 # Feature engineering pipeline
│   ├── train.py                         # Model training and OOT split
│   ├── evaluate.py                      # Diagnostic error analysis
│   └── predict.py                       # Prediction generator
│
├── models/
│   ├── pipeline.pkl                     # Fitted preprocessing pipeline
│   └── ensemble_model.pkl               # Fitted production ensemble
│
├── outputs/
│   ├── model_comparison.csv             # Model evaluation comparison
│   ├── feature_importances.png          # Feature importance chart
│   ├── residual_analysis.png            # Residual distribution plot
│   └── evaluation_report.txt            # Segment error breakdown
│
├── data/
│   └── december_chart_inputs.csv        # 31 December predictions
│
├── validation_predictions.csv           # 12,000 validation predictions
└── scorer_results/
    └── candidate_december.png           # Scorer generated chart
```

---

## Verification
Verification script passes cleanly:
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
Output:
- `Validated 12,000 final predictions.`
- `Validated 31 fixed December predictions.`
- `Created chart: scorer_results/candidate_december.png`
