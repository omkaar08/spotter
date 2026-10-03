# 🎙️ Loom Video Walkthrough Script (2–3 Minutes)

**Presenter:** Omkar Jagtap  
**Target Duration:** 2 minutes 30 seconds  
**Objective:** Present the technical solution for the Spotter AI Freight Rate Prediction Assessment clearly, concisely, and confidently.

---

## ⏱️ Video Structure & Talking Points

### 0:00 – 0:30 | Introduction & Problem Objective
> *"Hi team, I'm Omkar Jagtap. Today I'm presenting my machine learning solution for Spotter AI's Freight Rate Prediction Challenge.*
>
> *The objective of this challenge is to accurately predict spot rates in USD for 12,000 future freight loads in `validation.csv` and generate daily rate predictions for a fixed 360-mile Dry Van lane between Lexington and Fort Wayne in December 2025.*
>
> *Let's jump straight into the key data findings and technical approach."*

---

### 0:30 – 1:00 | Key EDA Findings & Data Quality Handling
> *"When exploring the 48,000 development loads, I uncovered four critical insights:*
>
> 1. **Haversine & Distance Integrity:** *Calculated Haversine distance from pickup and delivery coordinates correlates 0.9995 with actual road distance. The ratio between them gives us route circuitousness.*
> 2. **Quote Signal Physics:** *The product `distance * quote_signal` already explains 78% of rate variance. The remaining variance is driven by equipment type, market demand, and load weight density.*
> 3. **Equipment Premium:** *Reefer and Flatbed shipments command an 8% to 12% rate per mile premium over standard Dry Vans.*
> 4. **Addressing Missing Data:** *We handled missing values in load weight and market index using group medians by equipment and date, completely avoiding arbitrary mean imputation."*

---

### 1:00 – 1:35 | Data Split Strategy & Data Leakage Prevention
> *"To prevent data leakage and look-ahead bias, I established an **Out-Of-Time (OOT) Temporal Validation Split**.*
>
> *Instead of random K-Fold cross-validation—which causes look-ahead leakage in time-series data—I trained candidate models on Months 1 to 8 (Jan–Aug 2025) and validated them on Months 9 to 10 (Sept–Oct 2025).*
>
> *This strictly mirrors Spotter's official test setup where validation loads cover future months (Nov–Dec 2025)."*

---

### 1:35 – 2:10 | Model Choice, Empirical Benchmark & Scorer Output
> *"I benchmarked Ridge Regression, Random Forest, XGBoost, LightGBM, CatBoost, and a Weighted Ensemble.*
>
> *CatBoost was our single strongest model with an out-of-time MAE of **$147.50** and an R² of **0.8350**.*
>
> *For final submission, I deployed a **Weighted Spotter Ensemble** blending CatBoost, LightGBM, and Ridge Regression to ensure maximum generalization across market shifts.*
>
> *Running `score.py` validated all 12,000 validation predictions and generated our December rate chart (`candidate_december.png`), perfectly capturing the weekly Thursday/Friday market demand peaks."*

---

### 2:10 – 2:30 | Codebase Walkthrough & Conclusion
> *"In the repository, everything is modularized in `src/`: `data_validation.py` for schema audits, `preprocessing.py` for pipeline transformations, `train.py` for training, and `predict.py` for generating compliant submissions.*
>
> *Thank you for reviewing my submission, and I look forward to discussing this solution further with the Spotter team!"*
