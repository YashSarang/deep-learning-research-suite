"""
Part 1: Classical Retrieval

Goal:
    - Encode queries and tools independently
    - Compute similarity between query and each tool
    - Retrieve top-k most relevant tools
    - Evaluate BM25, msmarco-MiniLM, UAE-large-v1
    - Report Recall@1 and Recall@5 for each method
"""

import json
import numpy as np
from tqdm import tqdm

# ───────────────────────────────────
# Data loading
# ───────────────────────────────────
def load_data():
    with open("data/test_queries.json", "r") as f:
        test_queries = json.load(f)
    with open("data/tools.json", "r") as f:
        tools = json.load(f)
    return test_queries, tools


# ───────────────────────────────────
# Evaluation helpers
# ───────────────────────────────────
def recall_at_k(ranked_tool_names, gold_tool_name, k):
    """Returns 1 if the gold tool is in the top-k of ranked_tool_names, else 0."""
    return int(gold_tool_name in ranked_tool_names[:k])


def evaluate(test_queries, tools, retriever_fn, method_name):
    """
    Evaluate a retriever function over all test queries.
    retriever_fn(query_text, tools) -> list of tool names ranked by relevance (descending).
    """
    tool_names = list(tools.keys())
    tool_descriptions = list(tools.values())

    r1_total, r5_total = 0, 0
    n = len(test_queries)

    for sample in tqdm(test_queries, desc=f"Evaluating {method_name}"):
        query = sample["text"]
        gold = sample["gold_tool_name"]
        ranked = retriever_fn(query, tool_names, tool_descriptions)
        r1_total += recall_at_k(ranked, gold, 1)
        r5_total += recall_at_k(ranked, gold, 5)

    recall1 = r1_total / n
    recall5 = r5_total / n
    print(f"\n[{method_name}]  Recall@1 = {recall1:.4f}  |  Recall@5 = {recall5:.4f}")
    return recall1, recall5


# ═══════════════════════════════════
# Method 1: BM25 (sparse retrieval)
# ═══════════════════════════════════
def build_bm25(tool_names, tool_descriptions):
    from rank_bm25 import BM25Okapi

    # Build corpus: each document = "tool_name tool_description" tokenised by whitespace
    corpus = []
    for name, desc in zip(tool_names, tool_descriptions):
        doc = f"{name} {desc}"
        corpus.append(doc.lower().split())

    bm25 = BM25Okapi(corpus)
    return bm25


def retrieve_bm25(query, tool_names, tool_descriptions, bm25_model=None):
    scores = bm25_model.get_scores(query.lower().split())
    ranked_indices = np.argsort(scores)[::-1]
    return [tool_names[i] for i in ranked_indices]


# ═══════════════════════════════════
# Method 2 & 3: Dense retrieval
# ═══════════════════════════════════
def build_dense_index(tool_names, tool_descriptions, model):
    """Encode all tools once and return normalised embeddings."""
    docs = [f"{name}: {desc}" for name, desc in zip(tool_names, tool_descriptions)]
    embeddings = model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
    return embeddings  # (num_tools, dim)


def retrieve_dense(query, tool_names, tool_descriptions, model=None, tool_embeddings=None):
    q_emb = model.encode([query], normalize_embeddings=True)  # (1, dim)
    scores = (q_emb @ tool_embeddings.T).flatten()             # (num_tools,)
    ranked_indices = np.argsort(scores)[::-1]
    return [tool_names[i] for i in ranked_indices]


# ═══════════════════════════════════
# Main
# ═══════════════════════════════════
if __name__ == "__main__":
    test_queries, tools = load_data()
    tool_names = list(tools.keys())
    tool_descriptions = list(tools.values())

    results = {}

    # ── BM25 ──────────────────────
    print("\n" + "=" * 60)
    print("BM25 (sparse retrieval)")
    print("=" * 60)
    bm25_model = build_bm25(tool_names, tool_descriptions)
    r1, r5 = evaluate(
        test_queries, tools,
        lambda q, tn, td: retrieve_bm25(q, tn, td, bm25_model=bm25_model),
        "BM25",
    )
    results["BM25"] = {"Recall@1": r1, "Recall@5": r5}

    # ── msmarco-MiniLM ────────────
    print("\n" + "=" * 60)
    print("msmarco-MiniLM (dense retrieval)")
    print("=" * 60)
    from sentence_transformers import SentenceTransformer

    minilm = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-6-v3")
    minilm_embs = build_dense_index(tool_names, tool_descriptions, minilm)
    r1, r5 = evaluate(
        test_queries, tools,
        lambda q, tn, td: retrieve_dense(q, tn, td, model=minilm, tool_embeddings=minilm_embs),
        "msmarco-MiniLM",
    )
    results["msmarco-MiniLM"] = {"Recall@1": r1, "Recall@5": r5}

    # ── UAE-large-v1 ──────────────
    print("\n" + "=" * 60)
    print("UAE-large-v1 (dense retrieval)")
    print("=" * 60)
    uae = SentenceTransformer("WhereIsAI/UAE-Large-V1")
    uae_embs = build_dense_index(tool_names, tool_descriptions, uae)
    r1, r5 = evaluate(
        test_queries, tools,
        lambda q, tn, td: retrieve_dense(q, tn, td, model=uae, tool_embeddings=uae_embs),
        "UAE-large-v1",
    )
    results["UAE-large-v1"] = {"Recall@1": r1, "Recall@5": r5}

    # ── Summary table ─────────────
    print("\n" + "=" * 60)
    print(f"{'Method':<20} {'Recall@1':>10} {'Recall@5':>10}")
    print("-" * 42)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['Recall@1']:>10.4f} {metrics['Recall@5']:>10.4f}")
    print("=" * 60)

    # Save to JSON for report generation
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/part1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Part 1 results saved to results/part1_results.json")
