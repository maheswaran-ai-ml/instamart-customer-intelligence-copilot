
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_FILE = Path("outputs/rfm_table.csv")
OUTPUT_FILE = Path("outputs/customer_segments.csv")

rfm = pd.read_csv(INPUT_FILE)

features = ["recency", "frequency", "monetary"]
X = rfm[features]

# Standardise RFM features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the final K-means model.
model = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["segment_id"] = model.fit_predict(X_scaled)

# Profile each segment.
profile = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .round(2)
)

# Segment IDs are arbitrary labels; name segments from their profiles.
def segment_name(row):
    if row["mean_monetary"] == profile["mean_monetary"].max():
        return "Premium"
    if row["mean_recency"] <= profile["mean_recency"].median() and (
        row["mean_frequency"] >= profile["mean_frequency"].median()
    ):
        return "Loyal"
    if row["mean_recency"] > profile["mean_recency"].median():
        return "At Risk"
    return "Regular"

profile["segment_name"] = profile.apply(segment_name, axis=1)

# Mark only the highest-mean-monetary segment as Premium.
premium_segment = profile["mean_monetary"].idxmax()
rfm["segment_name"] = rfm["segment_id"].map(profile["segment_name"])
rfm["is_premium"] = rfm["segment_id"].eq(premium_segment)

# PCA on the scaled RFM matrix.
pca = PCA(n_components=2)
pca.fit(X_scaled)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
].to_csv(OUTPUT_FILE, index=False)

print("Customer segments saved to:", OUTPUT_FILE)

print("\nSegment profile:")
print(profile.to_string())

print("\nPremium segment ID:", premium_segment)
print("Premium customers:", int(rfm["is_premium"].sum()))

print("\nPCA explained variance ratios:")
print(pca.explained_variance_ratio_.round(3))
print("Total explained variance:", round(pca.explained_variance_ratio_.sum(), 3))

print("\nSaved columns:")
print(rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
].columns.tolist())
