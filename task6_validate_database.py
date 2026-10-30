
import sqlite3
from pathlib import Path

import pandas as pd

DB_FILE = Path("data/instamart.db")

with sqlite3.connect(DB_FILE) as connection:
    customers = pd.read_sql_query("SELECT * FROM customers", connection)
    products = pd.read_sql_query("SELECT * FROM products", connection)
    orders = pd.read_sql_query("SELECT * FROM orders", connection)

    checks = {
        "Customers row count": len(customers) == 1200,
        "Products row count": len(products) == 40,
        "Orders row count": len(orders) == 12309,
        "Customer IDs are unique": customers["customer_id"].is_unique,
        "Product IDs are unique": products["product_id"].is_unique,
        "Order IDs are unique": orders["order_id"].is_unique,
        "Customer ages have no missing values": customers["age"].notna().all(),
        "Customer ages are between 18 and 60": (
            customers["age"].between(18, 60).all()
        ),
        "No missing order amounts": orders["order_amount"].notna().all(),
        "No pending orders": not orders["status"].eq("pending").any(),
        "Cancelled orders have missing delivery times": (
            orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"]
            .isna()
            .all()
        ),
        "Completed orders have delivery times": (
            orders.loc[orders["status"].eq("completed"), "delivery_minutes"]
            .notna()
            .all()
        ),
    }

failed = []

for name, passed in checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if not passed:
        failed.append(name)

if failed:
    raise AssertionError("Failed checks: " + ", ".join(failed))

print("\nALL CHECKPOINTS PASSED")
