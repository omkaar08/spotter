# Technical Approach & Methodology Document

**Candidate Name:** Omkar Mahajan  
**Role:** Machine Learning Engineer Candidate  
**Assessment:** Spotter AI Freight Rate Prediction Challenge  
**Date:** October 2026  

---

## 1. Problem Formulation & Business Context

In spot freight markets, logistics brokers must dynamically price truckload shipments (`posted_rate` in USD) based on volatile market conditions, lane-specific carrier capacity, equipment constraints, and macroeconomic rate indicators. 

The objective of this challenge is to build a robust, reproducible, and leak-free machine learning model to predict spot rates for 12,000 future loads (`validation.csv`) and 31 daily loads for a fixed lane during December 2025 (`december-chart-inputs.csv`).

### Target & Variables
- **Target Variable:** `posted_rate` (Continuous USD rate, range: $57.22 to $25,533.00, median: $2,030.76).
- **Core Features:** Origin/Destination city names, geographical coordinates (`lat`, `lon`), road distance (miles), equipment category (`Dry Van`, `Reefer`, `Flatbed`), load weight (lbs), date (`YYYY-MM-DD`), daily market demand index (`market_index`), and initial quote signal (`quote_signal`).

---

## 2. Exploratory Data Analysis & Domain Physics

Through exploratory data analysis on the 48,000 development loads, we identified four key domain properties governing spot freight rates:

1. **Distance as Primary Physics Scale:**
   - Road distance exhibits a 0.9085 linear correlation with posted rates.
   - Haversine spherical distance calculated from coordinates shows a **0.9995 correlation** with the given road distance.
   - The ratio `distance / Haversine_distance` acts as a measure of **route circuitousness** (detour factor around geographic obstacles such as mountain ranges or bodies of water).

2. **Quote Signal Physics (`distance * quote_signal`):**
   - The product `quote_rate = distance * quote_signal` represents the baseline quote rate per mile.
   - Evaluating `posted_rate / (distance * quote_signal)` yields a median ratio of **1.0039**, indicating that broker quote signals directly anchor posted rates.
   - Alone, this physics formula achieves an R² of 0.7824 without any machine learning model.

3. **Equipment Premium:**
   - Rate Per Mile (RPM) varies significantly across equipment types:
     - **Dry Van:** Mean RPM = $2.12
     - **Flatbed:** Mean RPM = $2.29 (+8.0% premium over Dry Van)
     - **Reefer:** Mean RPM = $2.38 (+12.3% premium over Dry Van)

4. **Market Index Weekly Cycle:**
   - Analysis of `market_index` across dates reveals a distinct 7-day weekly cycle: market demand peaks around Thursdays/Fridays (1.02 - 1.04) and troughs on Sundays/Mondays (0.83 - 0.84).

---

## 3. Data Leakage Audit & Validation Strategy

### Preventing Data Leakage
To maintain strict data integrity, we enforced the following safeguards:
- **No Target Leakage:** Features derived from `posted_rate` (such as historical Rate Per Mile target encodings) were computed out-of-fold during cross-validation and computed strictly on training data for validation transformation.
- **No Future Information Leakage:** Imputation statistics (e.g. median weight by equipment) were learned strictly from training periods.
- **Validation Isolation:** `validation.csv` targets are withheld by Spotter AI and were never used during model training or hyperparameter tuning.

### Temporal Validation Split Framework
Because freight rate forecasting is deployed out-of-time, standard random K-Fold cross-validation suffers from temporal look-ahead leakage. 

We established an **Out-Of-Time (OOT) Temporal Validation Framework**:
- **Development Training Period (Months 1–8):** Jan 01, 2025 to Aug 31, 2025 (38,477 loads).
- **Holdout Validation Period (Months 9–10):** Sept 01, 2025 to Oct 31, 2025 (9,523 loads).

This structure strictly evaluates whether models generalize to future time horizons, mirroring the official evaluation on Nov 01 – Dec 31, 2025 (`validation.csv`).

---

## 4. Feature Engineering Pipeline

We engineered 35 numerical features in `FreightFeaturePipeline` (`src/preprocessing.py`):

1. **Spatial & Geographic Features:**
   - `haversine_dist`: Spherical distance in miles between pickup and delivery coordinates.
   - `circuitousness`: `distance / (haversine_dist + 1e-5)` (route detour metric).
   - `bearing`: Compass direction angle (0°–360°) from pickup to delivery.
   - `mid_lat`, `mid_lon`: Geographical midpoint coordinates.

2. **Domain Freight Physics:**
   - `quote_rate`: `distance * quote_signal`
   - `market_quote_rate`: `distance * quote_signal * market_index`
   - `weight_per_mile`: `weight / (distance + 1.0)`
   - `weight_ton`: Weight converted to US tons.

3. **Temporal & Seasonal Features:**
   - `month`, `dayofweek`, `dayofyear`, `quarter`, `is_weekend`
   - Cyclical Sine/Cosine encodings: `sin_dayofweek`, `cos_dayofweek`, `sin_month`, `cos_month`, `sin_dayofyear`, `cos_dayofyear`.

4. **Target Encoding (Historical Rate Per Mile):**
   - Out-of-fold historical mean RPM by `lane` (`pickup_delivery`), `pickup` city, `delivery` city, and `equipment` category.

---

## 5. Model Benchmarking & Empirical Comparison

We trained seven distinct models under identical OOT temporal validation conditions.

### Benchmark Results (Evaluated on Sept–Oct 2025 Holdout):

| Model | MAE ($) | RMSE ($) | R² | MedAE ($) | Rationale |
|---|---|---|---|---|---|
| **CatBoost** | **$147.50** | **$619.90** | **0.8350** | **$60.57** | **Best single model.** Excellent handling of categorical lookups and complex feature interactions. |
| **Spotter Weighted Ensemble** | **$149.56** | **$625.04** | **0.8322** | **$64.07** | **Selected Production Model.** Blends CatBoost (55%), LightGBM (30%), and Ridge (15%) for robust generalization. |
| LightGBM | $157.61 | $646.88 | 0.8203 | $60.62 | Fast leaf-wise tree splitting; highly complementary to CatBoost. |
| XGBoost | $180.43 | $690.25 | 0.7954 | $72.78 | Depth-wise tree growth; slightly overfit on distance extremes. |
| Random Forest | $186.15 | $678.37 | 0.8024 | $60.30 | Non-linear tree baseline; higher variance. |
| Ridge Regression | $200.96 | $639.81 | 0.8242 | $127.83 | Linear baseline with L2 regularization. |
| Quote Signal Formula | $246.02 | $711.80 | 0.7824 | $60.44 | Non-ML domain baseline formula (`distance * quote_signal`). |

### Model Selection Rationale
While single CatBoost achieved slightly lower MAE ($147.50 vs $149.56), the **Spotter Weighted Ensemble** was selected for final submission. Ensembling tree-based gradient boosting (CatBoost + LightGBM) with regularized linear models (Ridge) reduces variance and guards against unexpected market regime shifts in future out-of-sample data.

---

## 6. December Chart Generation & Scorer Compliance

For `data/december_chart_inputs.csv`:
- All 31 days (Dec 01 to Dec 31, 2025) represent a fixed load from Lexington to Fort Wayne (360 miles, Dry Van, 32,000 lbs).
- Coordinates (`pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`) were mapped from our city lookup table.
- Daily `market_index` and `quote_signal` values were populated using daily validation statistics from `validation.csv`.
- Generated predictions were verified using `score.py`, producing `scorer_results/candidate_december.png` with 0 errors.

---

## 7. Production Architecture & Scalability

In a real-world freight broker environment, this model can be deployed as a microservice:

```
[Inbound Load Payload] ──► [FastAPI Gateway]
                                 │
                                 ▼
                     [FreightFeaturePipeline]
                    (Enriches coords & RPM lookups)
                                 │
                                 ▼
                    [Spotter Ensemble ONNX]
                    (Sub-20ms batch inference)
                                 │
                                 ▼
                   [Dynamic Pricing API Response]
```

### Production Safeguards
1. **Covariate Shift Monitoring:** Track daily distributions of `market_index` and `quote_signal` using Evidently AI / Prometheus.
2. **Automated Retraining:** Trigger weekly model retraining as new freight settlement data clears.
