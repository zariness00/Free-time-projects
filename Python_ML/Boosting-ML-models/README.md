# Car Evaluation Classification with AdaBoost

This project implements a binary classification model using **AdaBoost** with a decision stump as its base estimator. The model predicts car acceptability based on several categorical features such as buying price, maintenance cost, safety, and more.

---

## Dataset

- **Source:** [UCI Machine Learning Repository – Car Evaluation Data Set](https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data)
- **Target Variable:** `accep` — transformed into binary:
  - `0` = unacceptable
  - `1` = acceptable, good, or very good

---

## Features Used

Categorical features one-hot encoded:
- `buying`
- `maint`
- `doors`
- `persons`
- `lug_boot`
- `safety`

---

## Model

- **Base Model:** `DecisionTreeClassifier(max_depth=1)` (a decision stump)
- **Ensemble Method:** `AdaBoostClassifier(n_estimators=5)`
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

---

## Results

Performance on the test set:

Accuracy: 0.8574
Precision: 0.7247
Recall: 0.8377
F1 Score: 0.7771


Confusion matrix:
             predicted yes    predicted no
actual yes 129 25
actual no 49 316

##  Platform

This project is developed and run on the **Codecademy** learning platform.

