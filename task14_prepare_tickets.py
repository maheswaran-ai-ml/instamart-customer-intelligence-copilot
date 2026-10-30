
import os
import re
import pandas as pd
import mock_llm

RAW_FILE = "data/raw/tickets.csv"
OUTPUT_FILE = "data/clean/tickets_redacted.csv"
MAX_INPUT_TOKENS = 400

os.makedirs("data/clean", exist_ok=True)

# Read empty ticket text as an empty string
df = pd.read_csv(RAW_FILE, keep_default_na=False)

# Exact patterns and placeholders, in the required order
patterns = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")
]

counts = {name: 0 for name, _, _ in patterns}
redacted_texts = []
empty_flags = []
token_counts = []
long_flags = []

for original_text in df["text"]:
    text = str(original_text)

    # Empty check uses the original ticket text
    is_empty = not text.strip()

    # Token estimate also uses the original ticket text
    estimated_tokens = mock_llm.est_tokens(text)
    is_long = estimated_tokens > MAX_INPUT_TOKENS

    # Apply redaction patterns in the exact required order
    redacted = text
    for name, pattern, replacement in patterns:
        redacted, count = pattern.subn(replacement, redacted)
        counts[name] += count

    redacted_texts.append(redacted)
    empty_flags.append(is_empty)
    token_counts.append(estimated_tokens)
    long_flags.append(is_long)

df["is_empty"] = empty_flags
df["redacted_text"] = redacted_texts
df["estimated_tokens"] = token_counts
df["is_long"] = long_flags

# Verify that no personal information remains
remaining = {}
for name, pattern, _ in patterns:
    remaining[name] = sum(
        len(pattern.findall(text)) for text in df["redacted_text"]
    )

assert sum(remaining.values()) == 0, (
    f"Personal information remains after redaction: {remaining}"
)

df.to_csv(OUTPUT_FILE, index=False)

print("Ticket preparation completed.")
print(f"Total tickets: {len(df)}")
print(f"Empty tickets: {sum(empty_flags)}")
print(f"Long tickets: {sum(long_flags)}")
print(f"Email substitutions: {counts['email']}")
print(f"Phone substitutions: {counts['phone']}")
print(f"Order ID substitutions: {counts['order_id']}")
print(f"Total substitutions: {sum(counts.values())}")
print(f"Remaining personal information: {sum(remaining.values())}")
print(f"Saved: {OUTPUT_FILE}")
