
import sqlite3
from pathlib import Path

import pandas as pd

DB_FILE = Path("data/instamart.db")
SQL_FILE = Path("sql/queries.sql")

sql_text = SQL_FILE.read_text(encoding="utf-8")

# Split the SQL file into individual queries using semicolons.
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

with sqlite3.connect(DB_FILE) as connection:
    for number, query in enumerate(queries, start=1):
        print(f"\n{'=' * 60}")
        print(f"QUERY {number}")
        print("=" * 60)

        result = pd.read_sql_query(query, connection)
        print(result.to_string(index=False))
        print(f"\nRows returned: {len(result)}")

print(f"\nSuccessfully executed {len(queries)} SQL queries.")
