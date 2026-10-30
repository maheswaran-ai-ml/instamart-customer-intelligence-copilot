
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

# Load the three raw tables without modifying the original files.
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")

tables = {
    "customers": customers,
    "products": products,
    "orders": orders,
}

# 1. Shape, data types, and missing values for every column.
for name, df in tables.items():
    print(f"\n{'=' * 60}")
    print(f"TABLE: {name.upper()}")
    print("=" * 60)

    print("\nShape (rows, columns):", df.shape)

    print("\nData types:")
    print(df.dtypes.to_string())

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })
    print("\nMissing values by column:")
    print(missing.to_string())

# 2. Order audit figures.
numeric_amount = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
numeric_delivery = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

print(f"\n{'=' * 60}")
print("RAW ORDERS AUDIT")
print("=" * 60)
print("Order rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending rows:", orders["status"].eq("pending").sum())
print("Distinct order_id:", orders["order_id"].nunique())
print("Non-numeric order_amount entries:", numeric_amount.isna().sum())
print("Negative delivery_minutes:", (numeric_delivery < 0).sum())
print(
    "Blank delivery_minutes:",
    orders["delivery_minutes"].isna().sum()
)

# 3. Customer audit figures.
age = pd.to_numeric(customers["age"], errors="coerce")
city = customers["city"].astype("string").str.strip()
gender = customers["gender"].astype("string").str.strip()

print(f"\n{'=' * 60}")
print("RAW CUSTOMERS AUDIT")
print("=" * 60)
print("Blank age:", customers["age"].isna().sum())
print("Age values outside 10–100:", ((age < 10) | (age > 100)).sum())
print("Distinct raw city spellings:", customers["city"].nunique())
print(
    "Distinct non-blank raw gender spellings:",
    gender[gender.notna() & gender.ne("")].nunique()
)
print(
    "Blank gender:",
    (customers["gender"].isna() | gender.eq("")).sum()
)

print("\nAudit completed. Original raw CSV files were not changed.")
