
import pandas as pd
import sqlite3
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/clean")
OUTPUT_FILE = OUTPUT_DIR / "orders.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load orders and products.
orders = pd.read_csv(RAW_DIR / "orders.csv")
products = pd.read_csv(RAW_DIR / "products.csv")

# 2. Convert order timestamp to datetime.
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")

# 3. Remove exact duplicate rows.
orders = orders.drop_duplicates()

# 4. Keep the latest record for each order_id.
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(subset="order_id", keep="last").copy()

# 5. Convert amount to numeric.
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Join product details to calculate replacement amounts.
orders = orders.merge(
    products[["product_id", "category", "price_inr"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

# Fill invalid/missing amounts using price, quantity and discount.
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# 6. Convert delivery time to numeric and invalidate negatives.
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders.loc[
    orders["delivery_minutes"] < 0, "delivery_minutes"
] = float("nan")

# Cancelled orders must have no delivery time.
orders.loc[
    orders["status"] == "cancelled", "delivery_minutes"
] = float("nan")

# Fill missing completed-order delivery times using the median
# for that product category among completed orders.
completed_mask = orders["status"] == "completed"

category_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(category_medians)
)

# Remove temporary product columns.
orders = orders.drop(columns=["category", "price_inr"])

# Save timestamp in the requested format.
orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

orders.to_csv(OUTPUT_FILE, index=False)

# Validation output.
completed = orders["status"] == "completed"
cancelled = orders["status"] == "cancelled"

print("Cleaned orders saved to:", OUTPUT_FILE)
print("Rows:", len(orders))
print("Distinct order IDs:", orders["order_id"].nunique())

print("\nOrder status counts:")
print(orders["status"].value_counts().sort_index().to_string())

print("\nMissing order amounts:", orders["order_amount"].isna().sum())
print(
    "Missing delivery times for completed orders:",
    orders.loc[completed, "delivery_minutes"].isna().sum(),
)
print(
    "Missing delivery times for cancelled orders:",
    orders.loc[cancelled, "delivery_minutes"].isna().sum(),
)
print(
    "Completed order amount total:",
    round(orders.loc[completed, "order_amount"].sum(), 2),
)
