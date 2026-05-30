# ---------- Data loading helpers (Step 1) ----------


"""
What you must do in Step 1
A) Load provided vocab (≈25k)

Read vocab list directly from the dataset JSON keys (or a separate vocab field if present).

Build a Vocab object (from your utils) with <UNK> added only for Task 4 OOV, not for embedding training. 

2026_1S_CS728_PA1

B) Load filtered CC-News subset

Parse the JSON values into a list of documents (each document = title + article or “passage”).

Tokenize each document using your consistent tokenizer.

When building co-occurrence: skip tokens not in vocab (don’t map to UNK) to comply with “training must be performed only on provided vocabulary.” 

2026_1S_CS728_PA1

C) Load CoNLL-2003 via datasets==3.6.0

from datasets import load_dataset

ds = load_dataset("conll2003")

You’ll use it in Task 3 (CRF) and Task 4 (MLP). 
"""
# utils.py
from __future__ import annotations
from collections import defaultdict

import re
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Callable, Any
from scipy import sparse
import numpy as np

import json
from pathlib import Path


# -----------------------------
# Tokenization / Normalization
# -----------------------------

_PUNCT_RE = re.compile(r"[\u2010-\u2015]")  # unicode hyphen range
_WS_RE = re.compile(r"\s+")

# Splits words and keeps basic punctuation as separate tokens.
# This is intentionally simple and stable across tasks.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)


def normalize_text(text: str, lowercase: bool = True) -> str:
    """
    Normalize raw text to be consistent across tasks.
    - Optionally lowercases
    - Normalizes unicode hyphen-like characters to '-'
    - Collapses whitespace
    """
    if text is None:
        return ""
    text = text.strip()
    text = _PUNCT_RE.sub("-", text)
    text = _WS_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    return text


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize into a list of tokens. Keeps punctuation as its own tokens.
    """
    text = normalize_text(text, lowercase=lowercase)
    if not text:
        return []
    return _TOKEN_RE.findall(text)


# -----------------------------
# Vocabulary + OOV handling
# -----------------------------

@dataclass
class Vocab:
    """
    A simple vocab wrapper with optional UNK.
    """
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    unk_token: Optional[str] = "<UNK>"

    @classmethod
    def from_tokens(cls, tokens: Iterable[str], add_unk: bool = True, unk_token: str = "<UNK>") -> "Vocab":
        uniq = list(dict.fromkeys(tokens))  # preserves order
        token_to_id = {t: i for i, t in enumerate(uniq)}
        id_to_token = uniq[:]
        if add_unk and unk_token not in token_to_id:
            token_to_id[unk_token] = len(id_to_token)
            id_to_token.append(unk_token)
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, unk_token=(unk_token if add_unk else None))

    def __len__(self) -> int:
        return len(self.id_to_token)

    def has(self, token: str) -> bool:
        return token in self.token_to_id

    def get_id(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if self.unk_token is meaningfully_none(self.unk_token):
            raise KeyError(f"OOV token '{token}' but vocab has no UNK")
        return self.token_to_id[self.unk_token]  # type: ignore[index]

    def get_token(self, idx: int) -> str:
        return self.id_to_token[idx]


def meaningfully_none(x: Any) -> bool:
    return x is None


def longest_affix_match(token: str, vocab: Vocab, min_len: int = 3) -> Optional[str]:
    """
    Heuristic: find the longest prefix or suffix (>= min_len) that exists in vocab.
    Returns matched token or None.
    """
    n = len(token)
    # Try longest prefix
    for L in range(n, min_len - 1, -1):
        pref = token[:L]
        if vocab.has(pref):
            return pref
    # Try longest suffix
    for L in range(n, min_len - 1, -1):
        suf = token[n - L:]
        if vocab.has(suf):
            return suf
    return None


def subword_averaging_vector(
    token: str,
    vocab: Vocab,
    embedding_matrix: np.ndarray,
    min_len: int = 3,
    max_ngrams: int = 200,
) -> Optional[np.ndarray]:
    """
    If token is OOV, average the embeddings of character n-grams that exist in vocab.
    This requires your vocab to contain such n-grams, which it probably does NOT.
    Provided here for completeness; if none found, returns None.
    """
    # Character n-grams from length min_len..min_len+2
    grams: List[str] = []
    for n in range(min_len, min_len + 3):
        if len(token) < n:
            continue
        for i in range(0, len(token) - n + 1):
            grams.append(token[i: i + n])
            if len(grams) >= max_ngrams:
                break
        if len(grams) >= max_ngrams:
            break

    vecs = []
    for g in grams:
        if vocab.has(g):
            vecs.append(embedding_matrix[vocab.get_id(g)])
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0)


@dataclass
class OOVResolver:
    """
    Plug-in OOV resolution for Task 4.
    Default is UNK fallback (if vocab has <UNK>).
    """
    mode: str = "unk"  # "unk", "affix", "random", "zero"
    unk_token: str = "<UNK>"
    seed: int = 1337

    def resolve_id(self, token: str, vocab: Vocab) -> int:
        if vocab.has(token):
            return vocab.get_id(token)

        mode = self.mode.lower().strip()

        if mode == "unk":
            if vocab.has(self.unk_token):
                return vocab.get_id(self.unk_token)
            # if no UNK, fall back to raise
            raise KeyError(f"OOV token '{token}' but no UNK in vocab")

        if mode == "affix":
            m = longest_affix_match(token, vocab)
            if m is not None:
                return vocab.get_id(m)
            if vocab.has(self.unk_token):
                return vocab.get_id(self.unk_token)
            raise KeyError(f"OOV token '{token}', no affix match, no UNK")

        if mode == "zero":
            # Caller should detect this and use zero vector. We return -1 sentinel.
            return -1

        if mode == "random":
            # Caller should detect this and use a deterministic random vector. Return -2 sentinel.
            return -2

        raise ValueError(f"Unknown OOV mode: {self.mode}")


def get_embedding_for_token(
    token: str,
    vocab: Vocab,
    embedding_matrix: np.ndarray,
    resolver: Optional[OOVResolver] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Fetch embedding for token with OOV strategy.
    - If resolver returns -1 => zero vector
    - If resolver returns -2 => random vector
    """
    if resolver is None:
        resolver = OOVResolver(
            mode="unk", unk_token=vocab.unk_token or "<UNK>")

    idx = resolver.resolve_id(token, vocab)
    d = embedding_matrix.shape[1]

    if idx >= 0:
        return embedding_matrix[idx]

    if idx == -1:
        return np.zeros((d,), dtype=embedding_matrix.dtype)

    if idx == -2:
        if rng is None:
            rng = np.random.default_rng(resolver.seed)
        # small random vector
        return rng.normal(loc=0.0, scale=0.01, size=(d,)).astype(embedding_matrix.dtype)

    raise RuntimeError("Unexpected OOV sentinel")


# -----------------------------
# Co-occurrence builder (sparse)
# -----------------------------


def build_cooccurrence_matrix(
    docs_tok,
    vocab,
    window_size: int = 5,
    symmetric: bool = True,
    distance_weighting: bool = False,
    dtype=np.float32,
):
    """
    Memory-safe co-occurrence builder.

    Instead of constructing a huge COO with many duplicate (i,j) entries and then calling
    sum_duplicates() (which can OOM), this aggregates counts in a dict keyed by (i,j).
    """
    V = len(vocab.id_to_token)
    w = int(window_size)

    # Map tokens to ids once per doc, and aggregate into dict
    counts = defaultdict(float)  # (i, j) -> count

    unk_id = vocab.token_to_id.get("<UNK>", None)

    for doc in docs_tok:
        # doc is list[str] tokens
        # If docs_tok already contains only in-vocab tokens, unk_id won't be used.
        ids = []
        for t in doc:
            tid = vocab.token_to_id.get(t)
            if tid is None:
                if unk_id is None:
                    continue
                tid = unk_id
            ids.append(tid)

        n = len(ids)
        for center in range(n):
            i = ids[center]
            left = max(0, center - w)
            right = min(n, center + w + 1)

            for ctx in range(left, right):
                if ctx == center:
                    continue
                j = ids[ctx]

                if distance_weighting:
                    dist = abs(ctx - center)
                    weight = 1.0 / float(dist)
                else:
                    weight = 1.0

                counts[(i, j)] += weight
                if symmetric:
                    counts[(j, i)] += weight

    # Convert dict to CSR sparse matrix
    nnz = len(counts)
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=dtype)

    for k, ((i, j), v) in enumerate(counts.items()):
        rows[k] = i
        cols[k] = j
        data[k] = v

    X = sparse.csr_matrix((data, (rows, cols)), shape=(V, V), dtype=dtype)
    # No huge duplicates exist now; sum_duplicates is cheap but optional
    X.sum_duplicates()
    return X


"""
Original co-occurrence builder without distance weighting, for reference:

def build_cooccurrence_matrix(
    documents: Iterable[List[str]],
    vocab: Vocab,
    window_size: int = 5,
    lowercase: bool = True,
    symmetric: bool = True,
) -> sparse.csr_matrix:
    
    # Build a word-word co-occurrence matrix X (|V| x |V|) from tokenized documents.
    # Counts how many times context word j appears within +/- window_size of center word i.

    # Notes:
    # - Only tokens present in vocab are counted; OOV are ignored (or mapped via vocab UNK if you pre-map them).
    # - Returns CSR matrix for fast row access and downstream operations (GloVe, SVD).
    
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    vsize = len(vocab)

    for doc in documents:
        if not doc:
            continue
        # Optionally normalize tokens (useful if upstream skipped)
        toks = [t.lower() for t in doc] if lowercase else doc

        # Convert to vocab ids; skip tokens not in vocab (for training embeddings, per instructions)
        ids: List[int] = []
        for t in toks:
            if vocab.has(t):
                ids.append(vocab.get_id(t))
            # IMPORTANT: For embedding training, we do NOT want to introduce UNK co-occurrences
            # unless you explicitly decide so. So we simply skip.

        n = len(ids)
        for i in range(n):
            wi = ids[i]
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            for j in range(left, right):
                if j == i:
                    continue
                wj = ids[j]
                rows.append(wi)
                cols.append(wj)
                data.append(1.0)
                if symmetric:
                    rows.append(wj)
                    cols.append(wi)
                    data.append(1.0)

    X = sparse.coo_matrix((data, (rows, cols)), shape=(vsize, vsize), dtype=np.float32)
    X.sum_duplicates()
    return X.tocsr()
"""

# -----------------------------
# Nearest neighbors (cosine)
# -----------------------------


def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, eps, None)


def top_k_cosine_neighbors(
    query_word: str,
    vocab: Vocab,
    embeddings: np.ndarray,
    k: int = 5,
    exclude_self: bool = True,
    resolver: Optional[OOVResolver] = None,
) -> List[Tuple[str, float]]:
    """
    Returns [(neighbor_word, cosine_sim), ...] of top-k cosine neighbors.
    Assumes embeddings is shape (|V|, d).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array (|V|, d)")

    # Normalize for cosine via dot product
    E = l2_normalize_rows(embeddings.astype(np.float32, copy=False))

    if resolver is None:
        resolver = OOVResolver(
            mode="unk", unk_token=vocab.unk_token or "<UNK>")

    qid = resolver.resolve_id(query_word, vocab)
    if qid < 0:
        # If query is OOV and resolver returns zero/random sentinel, we can’t define neighbors well.
        raise KeyError(
            f"Query word '{query_word}' not in vocab and cannot be resolved to an id.")

    qvec = E[qid]
    sims = E @ qvec  # (|V|,)

    if exclude_self:
        sims[qid] = -np.inf

    top_idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return [(vocab.get_token(i), float(sims[i])) for i in top_idx]


# -----------------------------
# Metrics (NER)
# -----------------------------

def token_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if len(y_true) == 0:
        return 0.0
    correct = 0
    for a, b in zip(y_true, y_pred):
        if a == b:
            correct += 1
    return correct / len(y_true)


def macro_f1_from_int_labels(y_true: List[int], y_pred: List[int], num_classes: Optional[int] = None) -> float:
    """
    Macro-F1 over integer class labels.
    No dependency on sklearn; safe for restricted environments.
    """
    if len(y_true) == 0:
        return 0.0

    if num_classes is None:
        num_classes = int(max(max(y_true, default=0),
                          max(y_pred, default=0)) + 1)

    # per-class TP/FP/FN
    tp = np.zeros((num_classes,), dtype=np.int64)
    fp = np.zeros((num_classes,), dtype=np.int64)
    fn = np.zeros((num_classes,), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    f1s: List[float] = []
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              ) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return float(np.mean(f1s))


def flatten_ner_sequences(
    y_true_seqs: List[List[int]],
    y_pred_seqs: List[List[int]],
    ignore_label: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """
    Flattens list-of-sequences into a single list for metric computation.
    Optionally ignores a label id (e.g., padding).
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    for ts, ps in zip(y_true_seqs, y_pred_seqs):
        m = min(len(ts), len(ps))
        for i in range(m):
            if ignore_label is not None and ts[i] == ignore_label:
                continue
            y_true.append(ts[i])
            y_pred.append(ps[i])
    return y_true, y_pred


# -----------------------------
# Small helpers for reproducibility/logging
# -----------------------------

def set_global_seed(seed: int = 1337) -> None:
    np.random.seed(seed)


def batch_iterable(xs: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), batch_size):
        yield xs[i: i + batch_size]


def load_ccnews_json(path: str) -> dict:
    """
    Loads the provided CC-News filtered JSON.
    JSON structure per assignment:
    keys = vocabulary tokens
    values = list of index (0-based) and corresponding passages from CC News.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_vocab_from_ccnews_json(payload: dict) -> List[str]:
    """
    Returns vocab tokens from the JSON keys.
    """
    # Keys are vocab tokens per assignment statement.
    return list(payload.keys())


def _extract_passages_from_value(value: Any) -> List[str]:
    """
    Robustly extract passage strings from the JSON value.
    The assignment says: 'values are list of index (0 based) and corresponding passages'
    but format may vary slightly. This handles common patterns:
      - value = [{"idx": 0, "passage": "..."} , ...]
      - value = [[0, "..."], [12, "..."], ...]
      - value = [{"index": 0, "text": "..."} , ...]
      - value = ["...","..."] (unlikely, but safe)
    """
    passages: List[str] = []
    if value is None:
        return passages

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                passages.append(item)
            elif isinstance(item, dict):
                # common keys
                for k in ("passage", "text", "doc", "document", "content"):
                    if k in item and isinstance(item[k], str):
                        passages.append(item[k])
                        break
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
                passages.append(item[1])
    elif isinstance(value, dict):
        # sometimes nested formats
        for k in ("passages", "docs", "documents", "data"):
            if k in value and isinstance(value[k], list):
                passages.extend(_extract_passages_from_value(value[k]))
                break
    return passages


def iter_ccnews_documents(payload: dict) -> Iterator[str]:
    """
    Yields raw documents/passages from the CC-News payload.
    IMPORTANT: The JSON is keyed by vocab token; passages may repeat across keys.
    We deduplicate by hash to avoid training overweight duplicates.
    """
    seen = set()
    for _, v in payload.items():
        passages = _extract_passages_from_value(v)
        for doc in passages:
            doc = doc.strip()
            if not doc:
                continue
            h = hash(doc)
            if h in seen:
                continue
            seen.add(h)
            yield doc


def load_ccnews_documents_as_tokens(
    json_path: str,
    vocab: Vocab,
    lowercase: bool = True,
) -> List[List[str]]:
    """
    Loads CC-News subset and returns tokenized documents.
    Tokens not in vocab are kept here (for debugging / stats),
    but co-occurrence builder will skip them for embedding training.
    """
    payload = load_ccnews_json(json_path)
    docs_tok: List[List[str]] = []
    for doc in iter_ccnews_documents(payload):
        toks = tokenize(doc, lowercase=lowercase)
        docs_tok.append(toks)
    return docs_tok


def compute_oov_rate_in_docs(docs_tok: List[List[str]], vocab: Vocab) -> float:
    total = 0
    oov = 0
    for doc in docs_tok:
        for t in doc:
            total += 1
            if not vocab.has(t):
                oov += 1
    return (oov / total) if total > 0 else 0.0


# ---------- CoNLL-2003 loader (Task 3/4) ----------
def load_conll2003():
    """
    Loads CoNLL-2003 dataset using HuggingFace datasets.
    Requirement: datasets==3.6.0 (per assignment).
    Returns the DatasetDict.
    """
    from datasets import load_dataset
    return load_dataset("conll2003", trust_remote_code=True)
