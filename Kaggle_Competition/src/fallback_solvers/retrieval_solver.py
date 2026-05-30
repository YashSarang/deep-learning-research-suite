"""
Retrieval Solver — FAISS + Sentence-Transformers offline RAG.
Queries a pre-built knowledge base index and scores options by semantic similarity.
"""
from __future__ import annotations

import pickle
import numpy as np
from src.decision.decision_engine import SolverResult

# ── Lazy singletons ───────────────────────────────────────────────────────────
_faiss_index = None
_faiss_metadata: list[str] = []
_embedder = None


def _load_index(faiss_path: str, metadata_path: str):
    """Load FAISS index and metadata. Cached after first call."""
    global _faiss_index, _faiss_metadata
    if _faiss_index is not None:
        return _faiss_index, _faiss_metadata

    try:
        import faiss
        _faiss_index = faiss.read_index(faiss_path)
        with open(metadata_path, "rb") as f:
            _faiss_metadata = pickle.load(f)
        print(f"[Retrieval] FAISS index loaded: {_faiss_index.ntotal} vectors")
    except Exception as e:
        print(f"[Retrieval] Failed to load FAISS index: {e}")
        _faiss_index = None
        _faiss_metadata = []

    return _faiss_index, _faiss_metadata


def _load_embedder(model_dir: str):
    """Load SentenceTransformer. Cached after first call."""
    global _embedder
    if _embedder is not None:
        return _embedder

    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_dir, device="cpu")  # Keep CPU, save GPU for VLM
        print("[Retrieval] Embedder loaded.")
    except Exception as e:
        print(f"[Retrieval] Failed to load embedder: {e}")
        _embedder = None

    return _embedder


def _embed(text: str, embedder) -> np.ndarray:
    """Embed a single string. Returns float32 numpy array."""
    vec = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def solve(extracted_data: dict, config: dict | None = None) -> SolverResult:
    """
    Retrieve relevant knowledge base chunks and score each answer option.

    Scoring:
      - Embed the question.
      - Retrieve top-K chunks from FAISS.
      - For each option, compute cosine similarity between option text
        and retrieved chunks.
      - Option with highest avg similarity score wins.

    Returns:
        SolverResult with answer ∈ {"1","2","3","4"} or None.
    """
    ans = SolverResult(answer=None, confidence=0.0)

    if config is None:
        return ans

    paths = config.get("paths", {})
    faiss_path    = paths.get("faiss_index", "")
    meta_path     = paths.get("faiss_metadata", "")
    embedder_path = paths.get("embedding_model_dir", "")

    index, metadata = _load_index(faiss_path, meta_path)
    embedder = _load_embedder(embedder_path)

    if index is None or embedder is None:
        return ans

    question = extracted_data.get("question", "")
    options  = extracted_data.get("options", {})

    if not question or not options:
        return ans

    # ── Retrieve top-K context chunks for the question ────────────────────
    try:
        q_vec = _embed(question, embedder)               # shape (1, D)
        k     = min(5, index.ntotal)
        distances, indices = index.search(q_vec, k)      # cosine scores (0-1 with normalised vecs)
        retrieved_chunks = [metadata[i] for i in indices[0] if i < len(metadata)]
    except Exception as e:
        print(f"[Retrieval] Search failed: {e}")
        return ans

    if not retrieved_chunks:
        return ans

    context = " ".join(retrieved_chunks)

    # ── Score each option ─────────────────────────────────────────────────
    option_scores: dict[str, float] = {}
    for opt_key, opt_text in options.items():
        if not opt_text:
            option_scores[opt_key] = 0.0
            continue
        try:
            combined = f"{question} {opt_text}"
            opt_vec  = _embed(combined, embedder)
            ctx_vec  = _embed(context, embedder)
            sim      = float(np.dot(opt_vec, ctx_vec.T))
            option_scores[opt_key] = sim
        except Exception:
            option_scores[opt_key] = 0.0

    if not option_scores:
        return ans

    best_opt  = max(option_scores, key=option_scores.get)
    best_sim  = option_scores[best_opt]
    threshold = config["thresholds"].get("retrieval_similarity", 0.80)

    # Only return if similarity is meaningfully above threshold
    if best_sim >= threshold:
        ans.answer     = str(best_opt)
        ans.confidence = best_sim

    return ans
