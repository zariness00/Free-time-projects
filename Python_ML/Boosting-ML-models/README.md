# Car Evaluation Classification with Boosting models

This project uses ensemble learning techniques to classify car acceptability (`unacc` vs. `acc`, `good`, `vgood`) based on categorical car features.

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

## Models Used

- **AdaBoostClassifier** with `DecisionTreeClassifier(max_depth=1)`
- **GradientBoostingClassifier** with `n_estimators=15`

- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

---

## Results

Performance on the test set(AdaBoosting):

Accuracy: 0.8574
Precision: 0.7247
Recall: 0.8377
F1 Score: 0.7771


Confusion Matrix (AdaBoost)

|               | Predicted Yes | Predicted No |
|---------------|---------------|---------------|
| Actual Yes    | 129           | 25            |
| Actual No     | 49            | 316           |


Performance on the test set(Gradient Boosting):

Accuracy: 0.8979
Precision: 0.7886
Recall: 0.8961
F1 Score: 0.8389


Confusion matrix(Gradient Boosting):

|               | Predicted Yes | Predicted No |
|---------------|---------------|---------------|
| Actual Yes    | 138           | 16            |
| Actual No     | 37            | 328           |


##  Platform

This project is developed and run on the **Codecademy** learning platform.

