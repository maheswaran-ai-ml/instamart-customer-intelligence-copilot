
import os
import re
import json
import time
import random
import pandas as pd
import mock_llm

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = "data/clean/tickets_redacted.csv"
SEGMENTS_FILE = "outputs/customer_segments.csv"

OUTPUT_DIR = "outputs"
BUDGET_DEMO_FILE = os.path.join(OUTPUT_DIR, "triage_budget_demo.csv")
FINAL_FILE = os.path.join(OUTPUT_DIR, "triage_results.csv")

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

ZERO_SHOT_PROMPT = """
CLASSIFY each customer support ticket into exactly one category.

Allowed categories: billing, delivery, technical, fraud.

Read the customer's message and identify the main issue.
Return a JSON-only reply in exactly this form:
{"category": "...", "reason": "..."}
Do not include markdown or any text outside the JSON.
"""

FEW_SHOT_PROMPT = """
CLASSIFY each customer support ticket into exactly one category.

Allowed categories: billing, delivery, technical, fraud.

Return a JSON-only reply in exactly this form:
{"category": "...", "reason": "..."}
Do not include markdown or any text outside the JSON.

EXAMPLES:
Example 1:
Customer: My invoice shows an incorrect amount.
Reply: {"category": "billing", "reason": "Incorrect invoice amount"}

Example 2:
Customer: Someone made an unauthorized transaction on my account.
Reply: {"category": "fraud", "reason": "Unauthorized transaction"}

Example 3:
Customer: The app crashes when I try to log in.
Reply: {"category": "technical", "reason": "Application login issue"}
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

tickets = pd.read_csv(INPUT_FILE, keep_default_na=False)
segments = pd.read_csv(SEGMENTS_FILE, keep_default_na=False)

required_segment_columns = {
    "customer_id", "segment_name", "is_premium"
}

assert required_segment_columns.issubset(set(segments.columns)), (
    "customer_segments.csv must contain customer_id, "
    "segment_name and is_premium"
)

# Convert premium values safely to boolean
def to_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}

segments["is_premium"] = segments["is_premium"].apply(to_bool)

segment_lookup = segments.set_index("customer_id").to_dict("index")

# --------------------------------------------------
# PRIVACY ASSERTIONS
# --------------------------------------------------

PRIVACY_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\b[6-9]\d{9}\b"),
    re.compile(r"ORD-\d{6}")
]

for text in tickets["redacted_text"]:
    for pattern in PRIVACY_PATTERNS:
        assert not pattern.search(str(text)), (
            "Personal information found in redacted ticket"
        )

# --------------------------------------------------
# OUTPUT VALIDATION
# --------------------------------------------------

def validate_response(reply):
    # First: try JSON validation
    try:
        parsed = json.loads(reply)

        if isinstance(parsed, dict):
            category = parsed.get("category")

            if isinstance(category, str) and category in CATEGORIES:
                return category, "classified_json"

        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):
        pass

    # Second: keyword validation
    words = set(re.findall(r"[a-z]+", str(reply).lower()))
    found = words.intersection(CATEGORIES)

    if len(found) == 1:
        return next(iter(found)), "classified_keyword"

    return "", "flagged_invalid_output"


# --------------------------------------------------
# RETRY FUNCTION
# --------------------------------------------------

def call_with_retry(messages, ticket_id):
    for retry_number in range(4):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if retry_number >= 3:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket={ticket_id} "
                f"retry={retry_number} wait={wait:.4f}s"
            )

            time.sleep(wait)

    raise RuntimeError("Unexpected retry failure")


# --------------------------------------------------
# TRIAGE PIPELINE
# --------------------------------------------------

def run_pipeline(system_prompt, token_budget=None, run_name="run"):

    # Reset immediately before each independent full run
    mock_llm.reset()

    results = []
    total_tokens = 0
    retry_safe_errors = 0

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        original_text = str(ticket["text"])
        redacted_text = str(ticket["redacted_text"])

        result = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": "",
            "category": "",
            "segment_name": "",
            "priority": "",
            "route": "",
            "tokens_used": 0
        }

        # Input guardrail 1: empty tickets
        if not original_text.strip():
            result["status"] = "skipped_empty"
            results.append(result)
            continue

        # Input guardrail 2: long tickets
        estimated = mock_llm.est_tokens(original_text)

        if estimated > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"
            results.append(result)
            continue

        # Token budget check before every model call
        if token_budget is not None and total_tokens >= token_budget:
            result["status"] = "budget_exhausted"
            results.append(result)
            continue

        # User content must be exactly the redacted ticket text
        for pattern in PRIVACY_PATTERNS:
            assert not pattern.search(redacted_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text}
        ]

        try:
            response = call_with_retry(messages, ticket_id)

        except mock_llm.TransientLLMError:
            # No permanent error is expected with this mock
            retry_safe_errors += 1
            result["status"] = "flagged_invalid_output"
            results.append(result)
            continue

        reply = response.choices[0].message.content or ""
        used_tokens = response.usage.total_tokens

        total_tokens += used_tokens

        category, status = validate_response(reply)

        result["status"] = status
        result["category"] = category
        result["tokens_used"] = used_tokens

        results.append(result)

    output = pd.DataFrame(results)

    print("\n" + "=" * 55)
    print(f"RUN: {run_name}")
    print("=" * 55)
    print("Total tickets:", len(output))
    print("Status counts:")
    print(output["status"].value_counts().to_string())
    print("Total model tokens:", total_tokens)

    classified = output[
        output["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    # Accuracy comparison against true categories
    comparison = classified.merge(
        tickets[["ticket_id", "true_category"]],
        on="ticket_id",
        how="left"
    )

    correct = (
        comparison["category"] == comparison["true_category"]
    ).sum()

    classified_count = len(comparison)
    accuracy = (
        correct / classified_count * 100
        if classified_count else 0
    )

    tokens_per_ticket = (
        total_tokens / classified_count
        if classified_count else 0
    )

    print(f"Classified tickets: {classified_count}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Tokens per classified ticket: {tokens_per_ticket:.2f}")

    print("Permanent retry failures:", retry_safe_errors)

    return output, {
        "classified": classified_count,
        "correct": int(correct),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_ticket": tokens_per_ticket,
        "comparison": comparison
    }


# --------------------------------------------------
# RUN 1: ZERO-SHOT COMPARISON
# --------------------------------------------------

zero_results, zero_metrics = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=None,
    run_name="ZERO-SHOT"
)

print("\nZERO-SHOT CONFUSION TABLE")

zero_comparison = zero_metrics["comparison"]

print(
    pd.crosstab(
        zero_comparison["true_category"],
        zero_comparison["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False
    )
)


# --------------------------------------------------
# RUN 2: FEW-SHOT COMPARISON
# --------------------------------------------------

few_results, few_metrics = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=None,
    run_name="FEW-SHOT"
)


# --------------------------------------------------
# RUN 3: ZERO-SHOT TOKEN BUDGET DEMO
# --------------------------------------------------

budget_results, budget_metrics = run_pipeline(
    ZERO_SHOT_PROMPT,
    token_budget=6000,
    run_name="ZERO-SHOT BUDGET DEMO"
)

budget_results.to_csv(BUDGET_DEMO_FILE, index=False)

exhausted_count = (
    budget_results["status"] == "budget_exhausted"
).sum()

print("\nBUDGET DEMO")
print("Token budget: 6000")
print("Budget exhausted tickets:", exhausted_count)
print("Saved:", BUDGET_DEMO_FILE)


# --------------------------------------------------
# RUN 4: FINAL FEW-SHOT ROUTING
# --------------------------------------------------

final_results, final_metrics = run_pipeline(
    FEW_SHOT_PROMPT,
    token_budget=100000,
    run_name="FINAL FEW-SHOT ROUTING"
)

# Apply routing rules using customer segments
for index, row in final_results.iterrows():

    status = row["status"]
    category = row["category"]
    customer_id = row["customer_id"]

    customer_segment = segment_lookup.get(customer_id, {})

    segment_name = customer_segment.get("segment_name", "")
    is_premium = customer_segment.get("is_premium", False)

    final_results.at[index, "segment_name"] = segment_name

    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    }:
        route = "human_review"
        priority = "NORMAL"

    elif category == "fraud":
        route = "security_team"
        priority = "HIGH"

    elif is_premium:
        route = "premium_desk"
        priority = "HIGH"

    else:
        route = "support_queue"
        priority = "NORMAL"

    final_results.at[index, "route"] = route
    final_results.at[index, "priority"] = priority

# Required output column order
final_columns = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

final_results = final_results[final_columns]
final_results.to_csv(FINAL_FILE, index=False)

print("\nFINAL ROUTING COUNTS")
print(final_results["route"].value_counts().to_string())

print("\nFINAL PRIORITY COUNTS")
print(final_results["priority"].value_counts().to_string())

print("\nFinal rows:", len(final_results))
print("Saved:", FINAL_FILE)

# --------------------------------------------------
# COMPARISON SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("ZERO-SHOT vs FEW-SHOT COMPARISON")
print("=" * 55)

summary = pd.DataFrame([
    {
        "prompt": "zero_shot",
        "classified": zero_metrics["classified"],
        "accuracy_pct": round(zero_metrics["accuracy"], 2),
        "total_tokens": zero_metrics["total_tokens"],
        "tokens_per_ticket": round(
            zero_metrics["tokens_per_ticket"], 2
        )
    },
    {
        "prompt": "few_shot",
        "classified": few_metrics["classified"],
        "accuracy_pct": round(few_metrics["accuracy"], 2),
        "total_tokens": few_metrics["total_tokens"],
        "tokens_per_ticket": round(
            few_metrics["tokens_per_ticket"], 2
        )
    }
])

print(summary.to_string(index=False))

