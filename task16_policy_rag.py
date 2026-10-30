
import csv
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "data" / "policies"
EVAL_FILE = BASE_DIR / "data" / "raw" / "policy_eval.csv"


def load_chunks(overlap=10):
    """Load policy files and split them into 40-word chunks."""
    chunks = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()

        if not words:
            continue

        step = 40 - overlap
        start = 0

        while start < len(words):
            chunk_words = words[start:start + 40]

            chunks.append({
                "source": path.name,
                "text": " ".join(chunk_words)
            })

            if start + 40 >= len(words):
                break

            start += step

    return chunks


def build_index(overlap=10):
    """Build a TF-IDF index for the selected chunk overlap."""
    chunks = load_chunks(overlap)

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    return chunks, vectorizer, matrix


# Main index uses the required 10-word overlap.
CHUNKS, VECTORIZER, TFIDF_MATRIX = build_index(overlap=10)


def retrieve(query, k=3, index=None):
    """Return the top-k chunks ranked by cosine similarity."""
    if index is None:
        chunks = CHUNKS
        vectorizer = VECTORIZER
        matrix = TFIDF_MATRIX
    else:
        chunks, vectorizer, matrix = index

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    ranked_indices = scores.argsort()[::-1][:k]

    return [
        {
            "source": chunks[i]["source"],
            "score": float(scores[i]),
            "text": chunks[i]["text"]
        }
        for i in ranked_indices
    ]


def answer(question, index=None):
    """Answer using only the three retrieved policy chunks."""
    results = retrieve(question, k=3, index=index)

    context_parts = []
    for item in results:
        context_parts.append(
            f"[{item['source']}]\n{item['text']}"
        )

    user_message = (
        "CONTEXT:\n"
        + "\n".join(context_parts)
        + "\nQUESTION:\n"
        + question
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer the question using only "
                "the supplied policy context. Do not use outside "
                "knowledge. If the information is not present in "
                "the context, say: I could not find that information "
                "in the policy documents."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = mock_llm.chat(messages)
    model_text = response.choices[0].message.content

    return {
        "answer": model_text,
        "sources": list(dict.fromkeys(
            item["source"] for item in results
        ))
    }


def get_column(fieldnames, candidates):
    """Find a CSV column while allowing common naming variations."""
    normalized = {
        re.sub(r"[^a-z0-9]", "", name.lower()): name
        for name in fieldnames
    }

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not find column. Expected one of {candidates}. "
        f"CSV columns are: {fieldnames}"
    )


def evaluate(index, label):
    """Evaluate retrieval, expected facts and unsupported questions."""
    with EVAL_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    question_col = get_column(
        fieldnames, ["question"]
    )
    expected_doc_col = get_column(
        fieldnames, ["expected_doc", "expected_document"]
    )
    expected_fact_col = get_column(
        fieldnames, ["expected_fact", "expected_answer"]
    )

    answerable_total = 0
    doc_hits = 0
    fact_hits = 0
    unsupported_total = 0
    fallback_hits = 0
    failures = []

    for row in rows:
        question = row[question_col].strip()
        expected_doc = (row.get(expected_doc_col) or "").strip()
        expected_fact = (row.get(expected_fact_col) or "").strip()

        results = retrieve(question, k=3, index=index)
        result = answer(question, index=index)
        answer_text = result["answer"] or ""

        # A blank expected document/fact marks an unsupported question.
        is_answerable = bool(expected_doc and expected_fact)

        if is_answerable:
            answerable_total += 1

            found_doc = any(
                item["source"] == expected_doc
                for item in results
            )
            found_fact = expected_fact in answer_text

            if found_doc:
                doc_hits += 1

            if found_fact:
                fact_hits += 1
            else:
                failures.append(question)

        else:
            unsupported_total += 1

            if answer_text.startswith("I could not find"):
                fallback_hits += 1

    print(f"\n--- {label} ---")
    print(f"Total questions: {len(rows)}")
    print(
        f"Expected document in top 3: "
        f"{doc_hits}/{answerable_total}"
    )
    print(
        f"Expected fact in answer: "
        f"{fact_hits}/{answerable_total}"
    )
    print(
        f"Unsupported questions with fallback: "
        f"{fallback_hits}/{unsupported_total}"
    )

    if failures:
        print("\nQuestions where expected fact was not found:")
        for question in failures:
            print("-", question)

    return {
        "label": label,
        "doc_hits": doc_hits,
        "answerable": answerable_total,
        "fact_hits": fact_hits,
        "fallback_hits": fallback_hits,
        "unsupported": unsupported_total,
        "failures": failures
    }


def main():
    print("PART D - POLICY RAG ASSISTANT")
    print("=" * 45)

    print(f"\nTotal chunks with 10-word overlap: {len(CHUNKS)}")

    source_counts = {}
    for chunk in CHUNKS:
        source_counts[chunk["source"]] = (
            source_counts.get(chunk["source"], 0) + 1
        )

    print("\nChunks per policy:")
    for source, count in source_counts.items():
        print(f"{source}: {count}")

    query = "refund policy late delivery"
    print(f"\nRetrieval test query: {query}")

    for number, item in enumerate(retrieve(query, k=3), 1):
        print(f"\nResult {number}")
        print("Source:", item["source"])
        print("Score:", round(item["score"], 4))
        print("Text:", item["text"])

    print("\nSample policy answer:")
    sample = answer(query)
    print("Answer:", sample["answer"])
    print("Sources:", sample["sources"])

    # Reset mock counters before each independent evaluation.
    mock_llm.reset()
    overlap_result = evaluate(
        (CHUNKS, VECTORIZER, TFIDF_MATRIX),
        "10-word overlap"
    )

    # Repeat with 40-word chunks and zero overlap.
    zero_index = build_index(overlap=0)

    mock_llm.reset()
    zero_result = evaluate(
        zero_index,
        "Zero overlap"
    )

    print("\n--- COMPARISON ---")
    print(
        "10-word overlap expected facts:",
        f"{overlap_result['fact_hits']}/"
        f"{overlap_result['answerable']}"
    )
    print(
        "Zero overlap expected facts:",
        f"{zero_result['fact_hits']}/"
        f"{zero_result['answerable']}"
    )

    if zero_result["failures"]:
        print("\nExample question affected by zero overlap:")
        print(zero_result["failures"][0])
        print(
            "Reason: the required information may be split across "
            "two adjacent chunks. Without overlap, the retriever "
            "may not return the complete information together."
        )

    print("\nTask 1 to Task 4 execution completed.")


if __name__ == "__main__":
    main()
