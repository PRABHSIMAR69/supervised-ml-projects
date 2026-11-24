# Supervised Machine Learning Projects

Two supervised ML projects built in Python using Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## 1. L&T Realty — Mumbai House Price Prediction (Regression)
- **Folder:** `MST_LTR/`
- **Dataset:** `Mumbai House Prices.csv` (76,038 raw rows cleaned to 54,892 after duplicate and 99th-percentile outlier removal)
- **Models:** Linear Regression (Baseline) vs. Random Forest Regressor (100 trees)
- **Results:**
  - Linear Regression: R2 = 0.8088 | RMSE = 60.67 Lakhs
  - Random Forest: R2 = 0.8638 | RMSE = 51.20 Lakhs

## 2. HDFC Bank — Loan Approval Prediction (Classification)
- **Folder:** `EST_HDFC/`
- **Dataset:** `loan_approval_dataset.csv` (4,269 applications, 11 features)
- **Models:** Logistic Regression (Baseline) vs. Random Forest Classifier (100 trees)
- **Results:**
  - Logistic Regression: 91.1% Accuracy
  - Random Forest: 98.8% Accuracy (854 test cases: 531 TP, 313 TN, 10 FP, 0 FN)
  - Top Driver: CIBIL Score (> 0.80 importance)
