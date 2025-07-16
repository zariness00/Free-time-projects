# Income Prediction Using Random Forest

This project applies a Random Forest classifier to predict whether a person's income exceeds $50K based on census data. It involves data preprocessing, model training, hyperparameter tuning, and feature importance analysis.

---

## Dataset

- **Source:** `adult.data` (Census Income dataset)
- **Columns Used:**
  - `age`
  - `workclass`
  - `marital-status`
  - `capital-gain`
  - `capital-loss`
  - `hours-per-week`
  - `sex`
  - `race`
  - Target: `income` (`<=50K` → 0, `>50K` → 1)

---

## Features & Processing

- **Categorical Encoding:** One-hot encoding applied to categorical features (`workclass`, `sex`, `race`, others)
- **Target Variable:** Binary classification (income ≤50K → 0, >50K → 1). 
- **Train-Test Split:** 80% training / 20% test set

---

## Model Training

- **Model:** RandomForestClassifier 
- **Initial Training:** Trained using default parameters
- **Evaluation Metric:** Accuracy score on test set

Accuracy score for default random forest: 82.282 %

## Hyperparameter Tuning
- **Parameter Tuned:** max_depth from 1 to 25

- **Metrics Tracked:** Accuracy on both training and test sets

- **Best Test Accuracy:** 84.370 %

- ***Max depth for maximum accuracy on test set:*** 18
- **Visualization: Accuracy vs. Max Depth plot shows the bias-variance tradeoff**
![Random Forest Accuracy Plot](https://github.com/user-attachments/assets/6dc37239-cfa8-453f-b4a8-5f7d0f0e7528)

## Feature Importance
Top 5 most important features as determined by the best-performing Random Forest model (max_depth=18):

| Feature | Importance |
| ------- | ---------- |
|capital-gain|0.228434|
|marital-status|0.215328|
|age|0.183806|
|hours-per-week|0.122719|
|capital-loss|0.083290|
