
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.dummy import DummyClassifier

DATA_FILE = Path("outputs/lapse_dataset.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

target = "lapsed"
drop_columns = ["customer_id", target]

X = df.drop(columns=drop_columns)
y = df[target]

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days",
]

categorical_features = [
    "favourite_category",
    "city",
    "gender",
]

assert X.isna().sum().sum() == 0, "Missing feature values found."
assert set(y.unique()) == {0, 1}, "Target must contain 0 and 1."

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            categorical_features,
        ),
    ],
    remainder="drop",
)

print("Total cohort:", len(df))
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Train lapsed:", int(y_train.sum()))
print("Test lapsed:", int(y_test.sum()))

# Baseline: always predict the majority class.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

print("\nBASELINE — majority class")
print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 4))
print("Lapsed-class F1:", round(f1_score(y_test, baseline_predictions, zero_division=0), 4))

# Logistic Regression: preprocessing is fitted only within each training fold.
lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

lr_search = GridSearchCV(
    estimator=lr_pipeline,
    param_grid={"classifier__C": [0.01, 0.1, 1, 10, 100]},
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

print("\nTraining Logistic Regression grid search...")
lr_search.fit(X_train, y_train)
lr_model = lr_search.best_estimator_
lr_predictions = lr_model.predict(X_test)

print("Best LR parameters:", lr_search.best_params_)
print("LR test accuracy:", round(accuracy_score(y_test, lr_predictions), 4))
print("LR test precision:", round(precision_score(y_test, lr_predictions, zero_division=0), 4))
print("LR test recall:", round(recall_score(y_test, lr_predictions, zero_division=0), 4))
print("LR test F1:", round(f1_score(y_test, lr_predictions, zero_division=0), 4))
print("\nLR classification report:")
print(classification_report(y_test, lr_predictions, zero_division=0))

# Random Forest: same leakage-safe preprocessing pipeline.
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(n_estimators=100, random_state=42),
        ),
    ]
)

rf_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid={
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10, 20],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

print("\nTraining Random Forest grid search...")
rf_search.fit(X_train, y_train)
rf_model = rf_search.best_estimator_
rf_predictions = rf_model.predict(X_test)

print("Best RF parameters:", rf_search.best_params_)
print("RF test accuracy:", round(accuracy_score(y_test, rf_predictions), 4))
print("RF test precision:", round(precision_score(y_test, rf_predictions, zero_division=0), 4))
print("RF test recall:", round(recall_score(y_test, rf_predictions, zero_division=0), 4))
print("RF test F1:", round(f1_score(y_test, rf_predictions, zero_division=0), 4))
print("\nRF classification report:")
print(classification_report(y_test, rf_predictions, zero_division=0))

# Save both tuned models for later analysis.
joblib.dump(lr_model, OUTPUT_DIR / "lapse_logistic_model.joblib")
joblib.dump(rf_model, OUTPUT_DIR / "lapse_random_forest_model.joblib")

# Save test-set predictions for comparison and later retention ranking.
comparison = pd.DataFrame({
    "customer_id": df.loc[X_test.index, "customer_id"].values,
    "actual_lapsed": y_test.values,
    "lr_prediction": lr_predictions,
    "lr_probability": lr_model.predict_proba(X_test)[:, 1],
    "rf_prediction": rf_predictions,
    "rf_probability": rf_model.predict_proba(X_test)[:, 1],
})
comparison.to_csv(OUTPUT_DIR / "lapse_test_predictions.csv", index=False)



# Oversampling experiment: duplicate minority-class rows in TRAIN only.
from sklearn.utils import resample

train_data = X_train.copy()
train_data["_target"] = y_train.to_numpy()

majority = train_data[train_data["_target"] == train_data["_target"].value_counts().idxmax()]
minority = train_data[train_data["_target"] != train_data["_target"].value_counts().idxmax()]

minority_oversampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42,
)

balanced_train = pd.concat([majority, minority_oversampled])
X_train_balanced = balanced_train.drop(columns="_target")
y_train_balanced = balanced_train["_target"]

oversampled_lr = lr_search.best_estimator_
oversampled_lr.fit(X_train_balanced, y_train_balanced)

oversampled_predictions = oversampled_lr.predict(X_test)

print("\nLOGISTIC REGRESSION — TRAINING-ONLY OVERSAMPLING")
print("Balanced training rows:", len(X_train_balanced))
print("Training class counts:")
print(y_train_balanced.value_counts().sort_index().to_dict())
print("Test precision:", round(precision_score(y_test, oversampled_predictions, zero_division=0), 4))
print("Test recall:", round(recall_score(y_test, oversampled_predictions, zero_division=0), 4))
print("Test F1:", round(f1_score(y_test, oversampled_predictions, zero_division=0), 4))

joblib.dump(oversampled_lr, OUTPUT_DIR / "lapse_model.joblib")

# Verify saved model reloads with identical predictions.
reloaded_model = joblib.load(OUTPUT_DIR / "lapse_model.joblib")
reloaded_predictions = reloaded_model.predict(X_test)
assert (reloaded_predictions == oversampled_predictions).all()

print("Saved and reloaded model predictions match exactly.")

print("\nSaved tuned models and test predictions in outputs/.")
