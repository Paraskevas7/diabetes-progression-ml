"""
Predicting diabetes disease progression from baseline measurements.

Dataset: scikit-learn's built-in diabetes dataset (442 patients, 10 baseline
features). Target: a quantitative measure of disease progression one year
after baseline.

Question: How well can baseline measurements predict disease progression,
and does a more flexible model beat a simple linear baseline?
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# 1. Load and inspect the data
data = load_diabetes(as_frame=True)
X, y = data.data, data.target
print(f"Samples: {X.shape[0]}, features: {X.shape[1]}")
print(f"Missing values: {int(X.isna().sum().sum())}")
print(f"Target range: {y.min():.0f} to {y.max():.0f} (mean {y.mean():.1f})\n")

# 2. Train/test split (test set is held out until the final evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 3. Candidate models
models = {
    "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE
    ),
}

# 4. Compare with 5-fold cross-validation on the training set only
print("Cross-validation on training data (R², higher is better):")
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    cv_results[name] = scores
    print(f"  {name:18s} mean {scores.mean():.3f}  (± {scores.std():.3f})")

# 5. Final evaluation on the held-out test set
print("\nHeld-out test set:")
rows = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
    })
results = pd.DataFrame(rows).set_index("Model").round(3)
print(results.to_string())
results.to_csv("results.csv")

# 6. Which features matter most? (linear model coefficients, standardized)
coefs = pd.Series(
    fitted["Linear regression"].named_steps["linearregression"].coef_,
    index=X.columns,
).sort_values(key=abs, ascending=False)
print("\nLinear regression coefficients (standardized):")
print(coefs.round(2).to_string())

# 7. Plot: predicted vs actual for the best model
best = results["R2"].idxmax()
best_pred = fitted[best].predict(X_test)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, best_pred, alpha=0.7)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
ax.plot(lims, lims, "--", color="gray", label="Perfect prediction")
ax.set_xlabel("Actual progression")
ax.set_ylabel("Predicted progression")
ax.set_title(f"{best}: predicted vs actual (test set)")
ax.legend()
fig.tight_layout()
fig.savefig("predicted_vs_actual.png", dpi=150)
print(f"\nBest model on test set: {best}. Plot saved to predicted_vs_actual.png")
