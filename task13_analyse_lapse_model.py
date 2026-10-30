
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs")

df = pd.read_csv(OUTPUT_DIR / "lapse_dataset.csv")
segments = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")
rf_model = joblib.load(OUTPUT_DIR / "lapse_random_forest_model.joblib")
lr_model = joblib.load(OUTPUT_DIR / "lapse_model.joblib")

target = "lapsed"
X = df.drop(columns=["customer_id", target])

# Random Forest feature importance after preprocessing.
preprocessor = rf_model.named_steps["preprocessor"]
classifier = rf_model.named_steps["classifier"]
feature_names = preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": classifier.feature_importances_,
}).sort_values("importance", ascending=False)

print("TOP 5 RANDOM FOREST FEATURES")
print(importance.head(5).to_string(index=False))

# Logistic Regression coefficient signs.
lr_preprocessor = lr_model.named_steps["preprocessor"]
lr_classifier = lr_model.named_steps["classifier"]
lr_names = lr_preprocessor.get_feature_names_out()

coefficients = pd.DataFrame({
    "feature": lr_names,
    "coefficient": lr_classifier.coef_[0],
})
coefficients["direction"] = np.where(
    coefficients["coefficient"] > 0,
    "positive (higher lapse odds)",
    "negative (lower lapse odds)",
)

print("\nLOGISTIC REGRESSION COEFFICIENTS")
for feature in [
    "numeric__recency",
    "numeric__frequency",
    "numeric__monetary",
]:
    row = coefficients.loc[coefficients["feature"] == feature]
    if not row.empty:
        print(row.to_string(index=False))

# Predict lapse probability for every eligible customer.
df["lapse_probability"] = lr_model.predict_proba(X)[:, 1]

# Join the Premium segment label from the existing segmentation output.
premium_ids = set(
    segments.loc[segments["is_premium"] == True, "customer_id"]
)

# Assignment requires the top 10 Premium customers from the test set.
test_predictions = pd.read_csv(OUTPUT_DIR / "lapse_test_predictions.csv")
premium_test = test_predictions[
    test_predictions["customer_id"].isin(premium_ids)
].copy()

premium_test = premium_test.sort_values(
    "lr_probability", ascending=False
).head(10)

retention = premium_test.rename(columns={
    "lr_probability": "lapse_probability",
    "actual_lapsed": "actually_lapsed",
})[["customer_id", "lapse_probability", "actually_lapsed"]]

retention.to_csv(OUTPUT_DIR / "retention_list.csv", index=False)

print("\nRETENTION LIST")
print("Premium customers in test set:", len(
    test_predictions[test_predictions["customer_id"].isin(premium_ids)]
))
print("Rows saved:", len(retention))
print(retention.to_string(index=False))
print("\nSaved outputs/retention_list.csv")
