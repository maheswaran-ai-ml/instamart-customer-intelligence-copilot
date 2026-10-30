
import pandas as pd
from pathlib import Path

RAW_FILE = Path("data/raw/customers.csv")
OUTPUT_DIR = Path("data/clean")
OUTPUT_FILE = OUTPUT_DIR / "customers.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load raw customer data.
customers = pd.read_csv(RAW_FILE)

# Standardise city spelling and remove surrounding spaces.
customers["city"] = customers["city"].str.strip().str.title()

# Standardise gender values.
gender_map = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female",
}

customers["gender"] = (
    customers["gender"]
    .fillna("")
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("unknown")
)

# Convert age to numeric and mark ages outside 10–100 as missing.
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[
    ~customers["age"].between(10, 100), "age"
] = float("nan")

# Fill missing ages with the median age of the same city.
city_median_age = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_age)

# Convert signup_date to datetime.
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="raise"
)

# Save cleaned customers.
customers.to_csv(OUTPUT_FILE, index=False)

# Print validation results.
print("Cleaned customers saved to:", OUTPUT_FILE)
print("\nCity counts:")
print(customers["city"].value_counts().sort_index().to_string())

print("\nGender counts:")
print(customers["gender"].value_counts().sort_index().to_string())

print("\nMissing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())
print("Distinct cities:", customers["city"].nunique())
