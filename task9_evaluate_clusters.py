
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

INPUT_FILE = Path("outputs/rfm_table.csv")
OUTPUT_FILE = Path("outputs/kmeans_evaluation.csv")

rfm = pd.read_csv(INPUT_FILE)

features = ["recency", "frequency", "monetary"]
X = rfm[features]

# Standardise RFM features before clustering.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []

# Evaluate K values from 2 through 8.
for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42,
    )
    labels = model.fit_predict(X_scaled)

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X_scaled, labels),
    })

evaluation = pd.DataFrame(results)
evaluation.to_csv(OUTPUT_FILE, index=False)

print("K-means evaluation saved to:", OUTPUT_FILE)
print(evaluation.round(3).to_string(index=False))

print("\nReference values:")
print("K=2: inertia 1820.8, silhouette 0.465")
print("K=3: inertia 1268.2, silhouette 0.445")
print("K=4: inertia 743.1, silhouette 0.526")
print("K=5: inertia 617.4, silhouette 0.481")
print("K=6: inertia 503.2, silhouette 0.482")
print("K=7: inertia 408.5, silhouette 0.416")
print("K=8: inertia 366.5, silhouette 0.412")
