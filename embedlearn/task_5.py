# task_5.py
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from utils import Vocab, top_k_cosine_neighbors, OOVResolver, get_embedding_for_token, token_accuracy, macro_f1_from_int_labels
from data_utils import load_ccnews_json, extract_vocab_from_ccnews_json, load_ccnews_documents_as_tokens, load_conll2003


# --------- Term-document matrix (same as Task 2) ----------
def build_term_document_matrix(docs_tok, vocab: Vocab, lowercase: bool = True) -> sparse.csr_matrix:
    rows, cols, data = [], [], []
    V = len(vocab)
    D = len(docs_tok)

    for doc_id, doc in enumerate(docs_tok):
        if not doc:
            continue
        toks = [t.lower() for t in doc] if lowercase else doc
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


# --------- TF-IDF transform on CSR term-doc ----------
def tfidf_transform_csr(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    X: (V, D) term-document counts in CSR.
    Returns TF-IDF weighted CSR, using smoothed IDF.
    """
    X = X.tocsr().astype(np.float32)

    V, D = X.shape
    # document frequency per term: number of docs where term appears
    # CSR row-wise: df[t] = nnz in that row
    df = np.diff(X.indptr).astype(np.float32)  # length V
    # smooth idf
    idf = np.log((D + 1.0) / (df + 1.0)) + 1.0  # length V

    # multiply each row by its idf: X_tfidf[t, :] = X[t, :] * idf[t]
    # efficient CSR row scaling:
    X_tfidf = X.copy()
    X_tfidf.data *= np.repeat(idf, np.diff(X_tfidf.indptr))
    return X_tfidf


# --------- SVD -> token embeddings UΣ ----------
def svd_USigma(X: sparse.csr_matrix, k: int, seed: int = 1337) -> np.ndarray:
    svd = TruncatedSVD(n_components=k, random_state=seed)
    svd.fit(X)
    Vt = svd.components_  # (k, D)
    E = (X @ Vt.T).astype(np.float32, copy=False)  # (V, k) = UΣ
    return E


def save_embeddings(path: str, vocab: Vocab, E: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=E, id_to_token=np.array(vocab.id_to_token, dtype=object))


# --------- Minimal MLP (same as Task 4) ----------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MLPTagger(nn.Module):
    def __init__(self, d_in: int, num_labels: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_labels),
        )

    def forward(self, x):
        return self.net(x)

def conll_split_to_xy(split, E: np.ndarray, vocab: Vocab, resolver: OOVResolver) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    rng = np.random.default_rng(resolver.seed)
    for ex in split:
        tokens = ex["tokens"]
        tags = ex["ner_tags"]
        for tok, tag in zip(tokens, tags):
            vec = get_embedding_for_token(tok, vocab=vocab, embedding_matrix=E, resolver=resolver, rng=rng)
            X_list.append(vec)
            y_list.append(int(tag))
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)

def train_eval_mlp(Xtr, ytr, Xva, yva, Xte, yte, d_in: int, num_labels: int, seed: int = 1337) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPTagger(d_in=d_in, num_labels=num_labels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=1024, shuffle=True)

    best_f1 = -1.0
    best_state = None

    for ep in range(1, 21):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(torch.from_numpy(Xva).to(device)), dim=1).cpu().numpy().tolist()
        yt = yva.tolist()
        f1 = macro_f1_from_int_labels(yt, pred, num_classes=num_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(torch.from_numpy(Xte).to(device)), dim=1).cpu().numpy().tolist()
    yt = yte.tolist()
    return {
        "test_acc": token_accuracy(yt, pred),
        "test_macro_f1": macro_f1_from_int_labels(yt, pred, num_classes=num_labels),
        "best_dev_macro_f1": best_f1,
    }


def main():
    # ---- set these ----
    CCNEWS_JSON = "data/cc_news_filtered.json"
    OUT_DIR = Path("outputs")
    ASSETS_DIR = Path("assets")

    # Use best SVD dimension from Task 4 (you must fill this in)
    best_d = 200

    # 5 diverse words for neighbor check (must be in vocab)
    check_words = ["city", "run", "music", "london", "market"]

    # OOV strategy must match what you used in Task 4 for fair comparison
    resolver = OOVResolver(mode="affix", unk_token="<UNK>", seed=1337)

    # ---- Load CC-News vocab + docs ----
    payload = load_ccnews_json(CCNEWS_JSON)
    vocab_tokens = extract_vocab_from_ccnews_json(payload)
    vocab = Vocab.from_tokens(vocab_tokens, add_unk=True)

    docs_tok = load_ccnews_documents_as_tokens(CCNEWS_JSON, vocab=vocab, lowercase=True)

    # ---- Build raw term-doc X ----
    X = build_term_document_matrix(docs_tok, vocab=vocab, lowercase=True)

    # ---- Raw SVD embeddings at best_d (for comparison) ----
    E_raw = svd_USigma(X, k=best_d, seed=1337)
    save_embeddings(str(OUT_DIR / f"svd_raw_USigma_d{best_d}.npz"), vocab, E_raw)

    # ---- TF-IDF transform + SVD embeddings ----
    X_tfidf = tfidf_transform_csr(X)
    E_tfidf = svd_USigma(X_tfidf, k=best_d, seed=1337)
    save_embeddings(str(OUT_DIR / f"svd_tfidf_USigma_d{best_d}.npz"), vocab, E_tfidf)

    # ---- Quality Check 1: neighbors raw vs tfidf ----
    qc_path = ASSETS_DIR / "task5_neighbors_raw_vs_tfidf.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    with qc_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word", "raw_neighbors", "tfidf_neighbors"])
        for word in check_words:
            raw_nn = top_k_cosine_neighbors(word, vocab=vocab, embeddings=E_raw, k=5)
            tf_nn = top_k_cosine_neighbors(word, vocab=vocab, embeddings=E_tfidf, k=5)
            w.writerow([word, str(raw_nn), str(tf_nn)])
            print(word)
            print("  raw  :", raw_nn)
            print("  tfidf:", tf_nn)

    # ---- Quality Check 2: retrain one final MLP using TF-IDF SVD vectors ----
    ds = load_conll2003()
    label_names = ds["train"].features["ner_tags"].feature.names
    num_labels = len(label_names)

    Xtr, ytr = conll_split_to_xy(ds["train"], E_tfidf, vocab, resolver)
    Xva, yva = conll_split_to_xy(ds["validation"], E_tfidf, vocab, resolver)
    Xte, yte = conll_split_to_xy(ds["test"], E_tfidf, vocab, resolver)

    metrics = train_eval_mlp(Xtr, ytr, Xva, yva, Xte, yte, d_in=best_d, num_labels=num_labels, seed=1337)
    print("\n[TF-IDF SVD MLP] ", metrics)

    # Save metrics for report
    metrics_path = ASSETS_DIR / "task5_tfidf_mlp_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["best_d", "oov_mode", "test_acc", "test_macro_f1", "best_dev_macro_f1"])
        w.writerow([best_d, resolver.mode, metrics["test_acc"], metrics["test_macro_f1"], metrics["best_dev_macro_f1"]])

    print(f"\nSaved neighbor comparison to: {qc_path}")
    print(f"Saved TF-IDF MLP metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
