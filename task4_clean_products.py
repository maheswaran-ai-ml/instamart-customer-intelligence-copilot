
import pandas as pd
from pathlib import Path

RAW_FILE = Path("data/raw/products.csv")
OUTPUT_DIR = Path("data/clean")
OUTPUT_FILE = OUTPUT_DIR / "products.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load product catalogue.
products = pd.read_csv(RAW_FILE)

# Standardise text fields.
for column in ["product_id", "product_name", "category"]:
    products[column] = products[column].astype("string").str.strip()

# Convert catalogue prices to numeric.
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="raise"
)

# Save clean products.
products.to_csv(OUTPUT_FILE, index=False)

print("Cleaned products saved to:", OUTPUT_FILE)
print("Rows:", len(products))
print("Distinct product IDs:", products["product_id"].nunique())
print("Missing values:", products.isna().sum().sum())
print("\nCategory counts:")
print(products["category"].value_counts().sort_index().to_string())
