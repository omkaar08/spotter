# 🎓 Spotter AI ML Engineering — Interview Preparation Guide

**Candidate Profile:** BE Information Technology Student | Aspiring Machine Learning & GenAI Engineer  
**Project Context:** Spot Freight Rate Prediction Pipeline  

---

## PART 1: 10 CORE PROJECT CONCEPTS EXPLAINED SIMPLY

### 1. What problem does this assessment solve?
Freight brokers need to quote accurate spot prices to shippers for moving truckloads of cargo. If quotes are too high, brokers lose business; if too low, brokers lose money paying carriers. This project predicts the expected spot rate (`posted_rate` in USD) using shipment attributes like origin, destination, distance, equipment type, load weight, date, and market indicators.

### 2. Why was the final model chosen?
We chose a **Weighted Ensemble** combining **CatBoost (55%)**, **LightGBM (30%)**, and **Ridge Regression (15%)**. CatBoost handles non-linear relationships and categorical lookup features best, LightGBM speeds up tree building, and Ridge Regression adds linear stability. Together, they reduced error to an out-of-time Mean Absolute Error (MAE) of **$149.56**, outperforming single baseline models.

### 3. How was the dataset cleaned?
We inspected all 48,000 training loads and 12,000 validation loads for schema consistency, missing values, duplicates, and out-of-bound coordinates. Missing values were confined to `weight` and `market_index`. We verified that city coordinates (`lat`, `lon`) were 1-to-1 unique for every city.

### 4. How were missing values handled?
- Missing `weight` values (~0.6% of data) were imputed using the median weight of the corresponding `equipment` type (e.g. Dry Van median weight ~32,000 lbs).
- Missing `market_index` values (~0.8% of data) were imputed using the daily median market index for that specific date in the dataset, preserving day-to-day market fluctuations.

### 5. What feature engineering was performed?
We created 35 numerical features across 4 categories:
1. **Spatial Physics:** Haversine spherical distance, compass bearing angle, and route circuitousness (`distance / Haversine_distance`).
2. **Freight Economics:** Baseline quote rate (`distance * quote_signal`), market quote rate (`distance * quote_signal * market_index`), and weight density (`weight / distance`).
3. **Temporal Seasonality:** Day of week, month, day of year, weekend flag, and cyclical sine/cosine transformations.
4. **Target Encoding:** Out-of-fold historical Rate Per Mile (RPM) for origin city, destination city, lane pair, and equipment type.

### 6. How was data leakage prevented?
- **No Temporal Leakage:** Models were evaluated on an Out-Of-Time (OOT) temporal split (Jan–Aug train, Sept–Oct validation), avoiding look-ahead bias inherent in random K-Fold splits.
- **No Preprocessing Contamination:** Imputation statistics and target encodings were computed strictly on training folds and applied to validation data without target exposure.

### 7. How do the evaluation metrics work?
- **MAE (Mean Absolute Error):** Measures average dollar error per prediction ($147.50).
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors ($619.90).
- **R² (Coefficient of Determination):** Measures proportion of rate variance explained by the model (0.8350 or 83.5%).
- **MedAE (Median Absolute Error):** Robust error metric representing typical middle error ($54.72).

### 8. What alternatives were considered?
We evaluated 7 distinct models:
- Formula Baseline (`distance * quote_signal`): MAE $246.02
- Ridge Regression: MAE $200.96
- Random Forest: MAE $186.15
- XGBoost: MAE $180.43
- LightGBM: MAE $157.61
- CatBoost: MAE $147.50
- Spotter Ensemble: MAE $149.56 (chosen for production stability).

### 9. What limitations remain?
- **Extreme Mileage Outliers:** On ultra-long hauls (>2,000 miles), rate per mile flattens non-linearly.
- **Unforeseen Market Shocks:** Macroeconomic fuel price spikes or severe weather events are not explicitly captured in historical dates.

### 10. How could the solution be improved in production?
- **Real-Time Feature Store:** Integrate live fuel price APIs (e.g. EIA diesel prices) and real-time weather alerts.
- **LLM/NLP Signals:** Extract qualitative carrier notes or traffic delay reports using lightweight NLP/embeddings.
- **ONNX Runtime:** Export models to ONNX for sub-10ms microservice inference.

---

## PART 2: 15 TECHNICAL INTERVIEW QUESTIONS & ANSWERS

#### Q1: Why did you use Haversine distance alongside given distance?
> **Answer:** Given distance is road miles, while Haversine is straight-line spherical distance. The ratio `given_distance / Haversine_distance` measures **route circuitousness**. A high ratio indicates winding mountain routes or detours around geographical obstacles, which directly increases fuel cost and driver time.

#### Q2: Why is a random 80/20 train/test split improper for freight rate forecasting?
> **Answer:** Freight rates vary across time due to seasonality and market cycles. A random 80/20 split leaks future date information into training folds (e.g. training on Dec 15 to predict Dec 14). Out-Of-Time (OOT) temporal splitting (Jan–Aug train, Sept–Oct val) evaluates genuine future predictive ability.

#### Q3: How did you handle target encoding without leaking data?
> **Answer:** Target encoding maps categorical features (like lane string) to mean historical Rate Per Mile. To prevent leakage, we computed target encodings out-of-fold during training, and smoothed sparse lanes using a global prior mean.

#### Q4: Why did CatBoost perform better than XGBoost on this dataset?
> **Answer:** CatBoost uses symmetric trees (oblivious decision trees) and native categorical target statistics, making it less prone to overfitting on high-cardinality categorical features like `pickup` and `delivery` cities.

#### Q5: What is the relationship between `posted_rate`, `distance`, and `quote_signal`?
> **Answer:** `quote_signal` represents the baseline quote rate per mile. Multiplying `distance * quote_signal` yields a baseline rate with a 0.898 correlation to `posted_rate` and an R² of 0.782 even without machine learning.

#### Q6: How did you ensure predictions in `validation_predictions.csv` match Spotter's requirements?
> **Answer:** We matched the exact shape (12,000 rows), column names (`load_id,predicted_rate`), ID ordering (`TE-000001` to `TE-012000`), clipped predictions to positive values, and validated execution via `score.py`.

#### Q7: Why did you include Ridge Regression in the ensemble alongside tree models?
> **Answer:** Tree models produce step-wise constant predictions and cannot extrapolate outside the bounding box of training feature ranges. Ridge Regression provides a smooth linear trend extrapolation baseline, stabilizing ensemble predictions on extreme miles or weights.

#### Q8: How did you fill missing market indices for the December chart inputs?
> **Answer:** `validation.csv` covers Nov 01 to Dec 31, 2025. We extracted the daily median market index from `validation.csv` for each day in December and mapped them to `december_chart_inputs.csv`.

#### Q9: What is the purpose of cyclical sine/cosine encodings for dates?
> **Answer:** Categorical numbers like day of week (0 to 6) or month (1 to 12) have artificial numerical jumps between Sunday (6) and Monday (0), or Dec (12) and Jan (1). Sine and cosine transformations map cyclic time continuously onto a 2D circle so the model understands proximity.

#### Q10: How would you handle rare or unseen pickup cities in production?
> **Answer:** Our feature pipeline defaults unseen cities to global mean Rate Per Mile (RPM) and uses regional latitude/longitude coordinates to infer spatial proximity to known logistics hubs.

#### Q11: What is the difference between MAE and RMSE in evaluating freight rates?
> **Answer:** MAE measures average absolute error in dollars, treating all errors linearly. RMSE squares errors before averaging, penalizing large outliers (like $2,000 misquotes) more heavily.

#### Q12: Why did you log-transform distance in linear models?
> **Answer:** Freight rate per mile decays with distance (tapering rate effect). Taking `log(distance)` linearizes this non-linear cost relationship for linear classifiers like Ridge.

#### Q13: How does `score.py` validate candidate outputs?
> **Answer:** `score.py` verifies file existence, column headers, exact row count (12,000), ID ordering, non-positive rate checks, and December date continuity (31 days), then plots `scorer_results/candidate_december.png`.

#### Q14: How would you monitor this model post-deployment?
> **Answer:** Monitor feature drift (Evidently AI), track prediction variance vs actual settled freight rates, and set automated alerts if daily MAE exceeds $180.

#### Q15: What was your biggest takeaway from building this solution?
> **Answer:** Domain physics features (`distance * quote_signal`) combined with temporal validation provide far greater performance gains than blindly tuning hyperparameters on raw features.
