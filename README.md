# Predicting Diabetes Disease Progression

A beginner machine learning project comparing linear models and a random forest on a clinical regression problem.

## Question

How well can baseline measurements predict diabetes disease progression one year later, and does a more flexible model beat a simple linear baseline?

## Dataset

scikit-learn's built-in diabetes dataset: 442 patients, 10 baseline features (age, sex, BMI, blood pressure, and six blood serum measurements), and a quantitative progression score as the target. The feature names are anonymized in the dataset, so the coefficient interpretation below is limited.

No missing values were found.

## Method

1. Split the data 80/20 into training and test sets (random seed 42).
2. Compare three models with 5-fold cross-validation on the training set only:
   - Linear regression (standardized features)
   - Ridge regression (standardized features, alpha = 1.0)
   - Random forest (300 trees, minimum 5 samples per leaf)
3. Evaluate each model once on the held-out test set.

## Results

Cross-validation R² (training data):

| Model | Mean R² | Std |
|---|---|---|
| Linear regression | 0.449 | 0.144 |
| Ridge regression | 0.451 | 0.139 |
| Random forest | 0.412 | 0.132 |

Test set:

| Model | R² | MAE |
|---|---|---|
| Linear regression | 0.453 | 42.79 |
| Ridge regression | 0.454 | 42.81 |
| Random forest | 0.459 | 43.36 |

## Interpretation

- All three models explain roughly 45% of the variance in progression. The remaining variation is likely driven by factors not in this dataset.
- The random forest had the highest test R², but the margin over linear models is tiny (0.006) and reversed in cross-validation. The sensible conclusion is that the extra complexity isn't justified here.
- In the linear model, the largest standardized coefficients are on blood serum features s1, s5, and BMI, followed by blood pressure. Several serum features are strongly correlated with each other, so individual coefficients should not be read as independent effects.

## Limitations

- Small dataset (442 rows), so test results vary with the split.
- Feature names are anonymized, and the data is a teaching dataset rather than a clinical study. These results should not be used for medical decisions.

## Reproducing

```bash
pip install scikit-learn pandas matplotlib
python analysis.py
```

The script prints all results and saves `results.csv` and `predicted_vs_actual.png`.
