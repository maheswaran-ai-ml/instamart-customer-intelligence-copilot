
from pathlib import Path

import pandas as pd

ORDERS_FILE = Path("data/clean/orders.csv")
CUSTOMERS_FILE = Path("data/clean/customers.csv")
OUTPUT_FILE = Path("outputs/lapse_dataset.csv")

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")


orders = pd.read_csv(ORDERS_FILE)
customers = pd.read_csv(CUSTOMERS_FILE)
products = pd.read_csv("data/clean/products.csv")

# Restore product category because cleaned orders don't contain it.
orders = orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Only completed orders before the cutoff can create behaviour features.
pre_cutoff = orders.loc[
    (orders["order_ts"] < CUTOFF)
    & (orders["status"] == "completed")
].copy()

# Cohort: customers with at least one completed order before cutoff.
last_orders = pre_cutoff.groupby("customer_id")["order_ts"].max()
cohort_ids = last_orders.index

features = (
    pre_cutoff.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
        favourite_category=("category", lambda s: s.mode().iloc[0]),
    )
)

# Cancellation rate uses all final-status orders before cutoff.
all_pre_cutoff = orders.loc[
    (orders["order_ts"] < CUTOFF)
    & orders["status"].isin(["completed", "cancelled"])
].copy()

status_counts = (
    all_pre_cutoff.groupby("customer_id")["status"]
    .agg(
        total_final_orders="size",
        cancelled_orders=lambda s: s.eq("cancelled").sum(),
    )
)
status_counts["cancellation_rate"] = (
    status_counts["cancelled_orders"] / status_counts["total_final_orders"]
)

features = features.join(status_counts[["cancellation_rate"]])
features["recency"] = (CUTOFF - features["last_order_ts"]).dt.days

# Merge customer demographics and tenure.
features = features.join(
    customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]]
)

features["tenure_days"] = (
    CUTOFF - features["signup_date"]
).dt.days

# Label: no completed order in September 2025 means lapsed.
september_completed = orders.loc[
    (orders["order_ts"] >= CUTOFF)
    & (orders["order_ts"] < LABEL_END)
    & (orders["status"] == "completed")
].groupby("customer_id")["order_id"].nunique()

features["lapsed"] = (
    ~features.index.isin(september_completed.index)
).astype(int)

features = features.reset_index()
features = features.drop(columns=["last_order_ts", "signup_date"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUTPUT_FILE, index=False)

print("Lapse dataset saved to:", OUTPUT_FILE)
print("Cohort customers:", len(features))
print("Lapsed customers:", int(features["lapsed"].sum()))
print("Lapsed percentage:", round(features["lapsed"].mean() * 100, 2), "%")
print("\nMissing values:")
print(features.isna().sum().to_string())
print("\nColumns:")
print(features.columns.tolist())
