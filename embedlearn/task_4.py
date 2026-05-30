# task_4.py
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import Vocab, OOVResolver, get_embedding_for_token, flatten_ner_sequences, token_accuracy, macro_f1_from_int_labels
from data_utils import load_conll2003


# -----------------------------
# Embedding loading
# -----------------------------

def load_embedding_npz(npz_path: str) -> Tuple[np.ndarray, List[str]]:
    z = np.load(npz_path, allow_pickle=True)
    E = z["embeddings"].astype(np.float32)
    id_to_token = list(z["id_to_token"])
    return E, id_to_token


def make_vocab_from_id_to_token(id_to_token: List[str]) -> Vocab:
    token_to_id = {t: i for i, t in enumerate(id_to_token)}
    # Detect UNK presence
    unk = "<UNK>" if "<UNK>" in token_to_id else None
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, unk_token=unk)


def ensure_unk_vector(E: np.ndarray, vocab: Vocab, unk_token: str = "<UNK>") -> Tuple[np.ndarray, Vocab]:
    """
    If <UNK> exists in vocab but its vector is untrained/unknown, you can set it to mean(E).
    If <UNK> doesn't exist, we append it and set vector to mean(E).
    """
    if vocab.has(unk_token):
        unk_id = vocab.get_id(unk_token)
        # Heuristic: if it's all zeros, replace with mean
        if np.allclose(E[unk_id], 0.0):
            E = E.copy()
            E[unk_id] = E.mean(axis=0)
        return E, vocab

    # Append UNK
    E2 = np.vstack([E, E.mean(axis=0, keepdims=True)]).astype(np.float32)
    id_to_token2 = vocab.id_to_token + [unk_token]
    vocab2 = make_vocab_from_id_to_token(id_to_token2)
    return E2, vocab2


# -----------------------------
# Dataset -> (X, y)
# -----------------------------

def conll_split_to_xy(
    split,
    E: np.ndarray,
    vocab: Vocab,
    resolver: OOVResolver,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []

    rng = np.random.default_rng(resolver.seed)

    for ex in split:
        tokens = ex["tokens"]
        tags = ex["ner_tags"]  # ints

        for tok, tag in zip(tokens, tags):
            vec = get_embedding_for_token(tok, vocab=vocab, embedding_matrix=E, resolver=resolver, rng=rng)
            X_list.append(vec)
            y_list.append(int(tag))

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# -----------------------------
# MLP model
# -----------------------------

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


def train_eval_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_dev: np.ndarray, y_dev: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    d_in: int,
    num_labels: int,
    seed: int = 1337,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
) -> Dict[str, float]:
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPTagger(d_in=d_in, num_labels=num_labels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_dev_f1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # dev eval
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_dev).to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        yt = y_dev.tolist()
        dev_acc = token_accuracy(yt, pred)
        dev_f1 = macro_f1_from_int_labels(yt, pred, num_classes=num_labels)

        # early keep best
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[ep {ep:02d}] dev_acc={dev_acc:.4f} dev_macroF1={dev_f1:.4f}")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # test eval
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()

    yt = y_test.tolist()
    test_acc = token_accuracy(yt, pred)
    test_f1 = macro_f1_from_int_labels(yt, pred, num_classes=num_labels)

    return {"test_acc": test_acc, "test_macro_f1": test_f1, "best_dev_macro_f1": best_dev_f1}


# -----------------------------
# Run all configs + save table
# -----------------------------

def main():
    ds = load_conll2003()
    label_names = ds["train"].features["ner_tags"].feature.names
    num_labels = len(label_names)

    # Must match Task1/Task2 dims
    d_list = [50, 100, 200, 300]  # replace with your exact list

    # Query words not needed here; this is NER evaluation
    out_csv = Path("assets") / "task4_mlp_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Choose and document OOV strategy in report
    resolver = OOVResolver(mode="affix", unk_token="<UNK>", seed=1337)

    # Where you saved embeddings
    glove_tmpl = "outputs/glove_d{d}.npz"
    svd_tmpl = "outputs/svd_USigma_d{d}.npz"

    rows = []
    for algo, tmpl in [("glove", glove_tmpl), ("svd", svd_tmpl)]:
        for d in d_list:
            path = tmpl.format(d=d)
            print(f"\n=== {algo.upper()}  d={d}  ({path}) ===")

            E, id_to_token = load_embedding_npz(path)
            vocab = make_vocab_from_id_to_token(id_to_token)
            E, vocab = ensure_unk_vector(E, vocab, unk_token="<UNK>")

            # Build X/y per split
            X_train, y_train = conll_split_to_xy(ds["train"], E, vocab, resolver)
            X_dev, y_dev = conll_split_to_xy(ds["validation"], E, vocab, resolver)
            X_test, y_test = conll_split_to_xy(ds["test"], E, vocab, resolver)

            metrics = train_eval_mlp(
                X_train, y_train,
                X_dev, y_dev,
                X_test, y_test,
                d_in=d,
                num_labels=num_labels,
                seed=1337,
                epochs=20,
                batch_size=1024,
                lr=1e-3,
            )

            row = {
                "algo": algo,
                "d": d,
                "oov_mode": resolver.mode,
                "test_acc": metrics["test_acc"],
                "test_macro_f1": metrics["test_macro_f1"],
                "best_dev_macro_f1": metrics["best_dev_macro_f1"],
            }
            rows.append(row)
            print("RESULT:", row)

    # Save table
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print best overall by test macro-F1
    best = max(rows, key=lambda r: r["test_macro_f1"])
    print("\nBEST OVERALL:", best)
    print(f"Saved results to: {out_csv}")


if __name__ == "__main__":
    main()
