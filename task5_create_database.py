
import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/clean")
DB_FILE = Path("data/instamart.db")

DB_FILE.parent.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")

# Create the SQLite database and load the cleaned tables.
with sqlite3.connect(DB_FILE) as connection:
    customers.to_sql("customers", connection, if_exists="replace", index=False)
    products.to_sql("products", connection, if_exists="replace", index=False)
    orders.to_sql("orders", connection, if_exists="replace", index=False)

    # Verify that all three tables exist and contain the expected rows.
    for table in ["customers", "products", "orders"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS row_count FROM {table}",
            connection,
        )["row_count"].iloc[0]
        print(f"{table}: {count} rows")

print(f"\nDatabase created successfully: {DB_FILE}")
