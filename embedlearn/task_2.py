# task_2.py
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from utils import Vocab, top_k_cosine_neighbors
from data_utils import load_ccnews_json, extract_vocab_from_ccnews_json, load_ccnews_documents_as_tokens


def build_term_document_matrix(docs_tok, vocab: Vocab, lowercase: bool = True) -> sparse.csr_matrix:
    """
    Build sparse term-document matrix X (|V| x |D|).
    X[t, d] = count of term t in document d.
    IMPORTANT: only count tokens present in provided vocab (skip OOV).
    """
    rows = []
    cols = []
    data = []

    V = len(vocab)
    D = len(docs_tok)

    for doc_id, doc in enumerate(docs_tok):
        if not doc:
            continue
        toks = [t.lower() for t in doc] if lowercase else doc

        # count in-doc term frequency
        tf = {}
        for t in toks:
            if vocab.has(t):
                tid = vocab.get_id(t)
                tf[tid] = tf.get(tid, 0) + 1

        for tid, cnt in tf.items():
            rows.append(tid)
            cols.append(doc_id)
            data.append(float(cnt))

    X = sparse.coo_matrix((data, (rows, cols)), shape=(V, D), dtype=np.float32)
    X.sum_duplicates()
    return X.tocsr()


def svd_embeddings_USigma(X: sparse.csr_matrix, k: int, seed: int = 1337) -> np.ndarray:
    """
    Returns E = U_k Σ_k using TruncatedSVD:
    E = X @ V_k  where V_k = (components_)^T
    """
    svd = TruncatedSVD(n_components=k, random_state=seed)
    svd.fit(X)
    Vt = svd.components_                # (k, D) = V_k^T
    # E = U Σ = X V
    E = X @ Vt.T                        # (V, k)
    E = E.astype(np.float32, copy=False)
    return E


def save_embeddings(path: str, vocab: Vocab, E: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=E, id_to_token=np.array(vocab.id_to_token, dtype=object))


def main():
    # ---- edit paths ----
    CCNEWS_JSON = "data/cc_news_filtered.json"
    OUT_DIR = "outputs"

    # ---- must match Task 1 dims ----
    d_list = [50, 100, 200, 300]  # replace with the exact set used in Task 1

    # ---- Load ----
    payload = load_ccnews_json(CCNEWS_JSON)
    vocab_tokens = extract_vocab_from_ccnews_json(payload)
    vocab = Vocab.from_tokens(vocab_tokens, add_unk=True)

    docs_tok = load_ccnews_documents_as_tokens(CCNEWS_JSON, vocab=vocab, lowercase=True)
    print(f"docs={len(docs_tok)} vocab={len(vocab)}")

    # ---- Build X (term-document) ----
    t0 = time.time()
    X = build_term_document_matrix(docs_tok, vocab=vocab, lowercase=True)
    tX = time.time() - t0
    print(f"[X built] shape={X.shape} nnz={X.nnz} time={tX:.2f}s")

    # ---- Query words must match Task 1 ----
    chosen_words = ["united", "city", "president"]  # replace with your 3 words

    for k in d_list:
        print(f"\n[SVD] k={k}")
        t1 = time.time()
        E = svd_embeddings_USigma(X, k=k, seed=1337)
        tE = time.time() - t1
        print(f"[E] shape={E.shape} time={tE:.2f}s")

        save_embeddings(f"{OUT_DIR}/svd_USigma_d{k}.npz", vocab, E)

        for qw in chosen_words:
            nn = top_k_cosine_neighbors(qw, vocab=vocab, embeddings=E, k=5, exclude_self=True)
            print(qw, "->", nn)


if __name__ == "__main__":
    main()
