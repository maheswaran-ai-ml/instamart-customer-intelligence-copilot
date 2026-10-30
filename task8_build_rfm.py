
from pathlib import Path

import pandas as pd

ORDERS_FILE = Path("data/clean/orders.csv")
OUTPUT_FILE = Path("outputs/rfm_table.csv")

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

# Load cleaned order data.
orders = pd.read_csv(ORDERS_FILE)
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Use completed orders only.
completed = orders.loc[orders["status"].eq("completed")].copy()

# Calculate RFM for every customer with at least one completed order.
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

# Recency is the number of whole days before the snapshot.
rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days

# Keep the requested RFM columns.
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].sort_values("customer_id").reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
rfm.to_csv(OUTPUT_FILE, index=False)

print("RFM table saved to:", OUTPUT_FILE)
print("Customers:", len(rfm))
print("\nRFM summary:")
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

print("\nReference checks:")
print("Recency median:", rfm["recency"].median())
print("Recency minimum:", rfm["recency"].min())
print("Recency maximum:", rfm["recency"].max())
print("Frequency mean:", round(rfm["frequency"].mean(), 2))
print("Frequency maximum:", rfm["frequency"].max())
print("Monetary mean:", round(rfm["monetary"].mean(), 2))
print("Monetary maximum:", round(rfm["monetary"].max(), 2))
