# task_1.py
import os
import time
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from utils import Vocab, build_cooccurrence_matrix, top_k_cosine_neighbors, set_global_seed
from data_utils import load_ccnews_json, extract_vocab_from_ccnews_json, load_ccnews_documents_as_tokens


def glove_weight(x: np.ndarray, x_max: float, alpha: float) -> np.ndarray:
    w = (x / x_max) ** alpha
    w = np.minimum(w, 1.0)
    return w


def train_glove_sgd(
    X: sparse.csr_matrix,
    d: int,
    lr: float,
    iters: int,
    x_max: float = 100.0,
    alpha: float = 0.75,
    seed: int = 1337,
    report_every: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Train GloVe using SGD over nonzero co-occurrence entries.
    Returns W, W_tilde, b, b_tilde, losses(list per iter).
    """
    set_global_seed(seed)

    Xcoo = X.tocoo()
    I = Xcoo.row.astype(np.int64)
    J = Xcoo.col.astype(np.int64)
    Xij = Xcoo.data.astype(np.float32)

    logX = np.log(Xij + 1e-12)
    wts = glove_weight(Xij, x_max=x_max, alpha=alpha).astype(np.float32)

    V = X.shape[0]
    W = 0.01 * np.random.randn(V, d).astype(np.float32)
    Wt = 0.01 * np.random.randn(V, d).astype(np.float32)
    b = np.zeros((V,), dtype=np.float32)
    bt = np.zeros((V,), dtype=np.float32)

    losses = []

    n = len(Xij)
    idx = np.arange(n)

    for it in range(1, iters + 1):
        t0 = time.time()
        np.random.shuffle(idx)

        total_loss = 0.0
        for k in idx:
            i = I[k]
            j = J[k]
            w = float(wts[k])
            target = float(logX[k])

            dot = float(np.dot(W[i], Wt[j]))
            pred = dot + float(b[i]) + float(bt[j])
            err = pred - target

            loss = w * (err * err)
            total_loss += loss

            g = 2.0 * w * err

            # --- NEW: gradient clipping to prevent exploding updates ---
            g = float(np.clip(g, -10.0, 10.0))

            Wi = W[i].copy()
            W[i] -= lr * (g * Wt[j])
            Wt[j] -= lr * (g * Wi)
            b[i] -= lr * g
            bt[j] -= lr * g

        avg_loss = total_loss / max(n, 1)
        losses.append(float(avg_loss))

        # --- NEW: stop early if we went unstable ---
        if not np.isfinite(avg_loss):
            print(
                f"[iter {it:03d}/{iters}] avg_loss={avg_loss} (non-finite). Aborting run (lr={lr}).")
            break

        # --- NEW: stop early if we went unstable ---
        if not np.isfinite(avg_loss):
            print(
                f"[iter {it:03d}/{iters}] avg_loss={avg_loss} (non-finite). Aborting run (lr={lr}).")
            break

        dt = time.time() - t0
        if report_every and (it % report_every == 0):
            print(
                f"[iter {it:03d}/{iters}] avg_loss={avg_loss:.6f}  time={dt:.2f}s  nnz={n}")

    return W, Wt, b, bt, losses


def save_loss_plot(losses: list[float], outpath: Path, title: str):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Avg Weighted Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(outpath))
    plt.close()


def save_embeddings(path: Path, vocab: Vocab, E: np.ndarray):
    """
    Save embeddings + id_to_token mapping needed for Task 4.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), embeddings=E, id_to_token=np.array(
        vocab.id_to_token, dtype=object))
    print(f"[saved] {path}")


def estimate_raw_pairs(docs_tok, window_size: int) -> int:
    """
    Rough estimate of raw (center, context) pairs generated before any dedup:
      approx sum(len(doc) * 2w).
    This is what causes gigantic COO + sum_duplicates OOM in some implementations.
    """
    w = int(window_size)
    return int(sum(len(doc) * (2 * w) for doc in docs_tok))


def main():
    # Make paths deterministic (relative paths depend on where you run from)
    BASE = Path(__file__).resolve().parent

    CCNEWS_JSON = BASE / "data" / "cc_news_filtered.json"
    ASSETS_DIR = BASE / "assets"
    OUT_DIR = BASE / "outputs"

    # ---- Hyperparam search space ----
    d_fixed = 200
    windows = [2, 5, 10]
    lrs = [0.025, 0.01, 0.005]
    iters = 30
    x_max = 100.0
    alpha = 0.75

    # Debug mode: build X once and exit (set env var GLOVE_BUILD_ONLY=1)
    BUILD_ONLY = os.getenv("GLOVE_BUILD_ONLY", "0") == "1"

    # Safety: skip very large windows if raw estimate is too big
    # You can tune this threshold based on your RAM. 1e8 is already huge.
    RAW_PAIR_LIMIT = int(os.getenv("GLOVE_RAW_PAIR_LIMIT", "120000000"))

    # ---- Load dataset + vocab ----
    payload = load_ccnews_json(str(CCNEWS_JSON))
    vocab_tokens = extract_vocab_from_ccnews_json(payload)
    vocab = Vocab.from_tokens(vocab_tokens, add_unk=True)

    docs_tok = load_ccnews_documents_as_tokens(
        str(CCNEWS_JSON), vocab=vocab, lowercase=True)

    # ---- Track results ----
    results_csv = ASSETS_DIR / "glove_search_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["w", "lr", "iters", "x_max", "alpha",
                        "final_loss", "total_train_s", "nnz"])

        best = None  # (final_loss, config_dict, model_params)

        for w in windows:
            est_pairs = estimate_raw_pairs(docs_tok, window_size=w)
            print(f"[preflight] w={w} approx_raw_pairs={est_pairs:,}")

            if est_pairs > RAW_PAIR_LIMIT:
                print(
                    f"[skip] w={w} estimated raw pairs too large for safe COO build (limit={RAW_PAIR_LIMIT:,}).")
                continue

            # build co-occurrence once per window
            t_build0 = time.time()
            try:
                X = build_cooccurrence_matrix(
                    docs_tok,
                    vocab=vocab,
                    window_size=w,
                    symmetric=True,
                    distance_weighting=False,
                )
            except np.core._exceptions._ArrayMemoryError as e:
                print(f"[OOM] w={w} failed during co-occurrence build: {e}")
                print(
                    "This indicates utils.build_cooccurrence_matrix is creating a massive COO and sum_duplicates() OOMs.")
                print(
                    "Fix is in utils.py: aggregate counts while building (no huge COO).")
                continue

            build_s = time.time() - t_build0
            print(f"[build] w={w} X.nnz={X.nnz} time={build_s:.2f}s")

            if BUILD_ONLY:
                print(
                    "[build-only] Exiting after building X (set GLOVE_BUILD_ONLY=0 to train).")
                return

            for lr in lrs:
                t0 = time.time()
                W, Wt, b, bt, losses = train_glove_sgd(
                    X=X,
                    d=d_fixed,
                    lr=lr,
                    iters=iters,
                    x_max=x_max,
                    alpha=alpha,
                    seed=1337,
                )
                train_s = time.time() - t0

                final_loss = float(losses[-1])
                if not np.isfinite(final_loss):
                    final_loss = float("inf")

                writer.writerow([w, lr, iters, x_max, alpha,
                                final_loss, train_s, int(X.nnz)])
                f.flush()

                plot_path = ASSETS_DIR / \
                    f"glove_loss_w{w}_d{d_fixed}_lr{lr}.png"
                save_loss_plot(
                    losses,
                    outpath=plot_path,
                    title=f"GloVe Loss (w={w}, d={d_fixed}, lr={lr})",
                )

                if (best is None) or (final_loss < best[0]):
                    best = (final_loss, {"w": w, "lr": lr, "iters": iters,
                            "x_max": x_max, "alpha": alpha}, (W, Wt, b, bt, X))

                print(
                    f"[done] w={w} lr={lr} final_loss={final_loss:.6f} train_time={train_s:.2f}s")

    if best is None:
        raise RuntimeError(
            "No configuration finished. If you saw OOM in co-occurrence build, you must fix utils.build_cooccurrence_matrix "
            "to avoid massive COO + sum_duplicates memory blow-up."
        )

    # ---- Best config ----
    best_loss, cfg, (W, Wt, b, bt, Xbest) = best
    print("BEST:", cfg, "loss:", best_loss)

    # ---- Final embedding ----
    E200 = W + Wt  # common choice

    # Save best 200-d embeddings FIRST (so you always get the file for Task 4)
    save_embeddings(OUT_DIR / "glove_best_d200.npz", vocab, E200)

    # Neighbor demo (safe)
    chosen_words = ["united", "city", "president"]
    for qw in chosen_words:
        if qw not in vocab.token_to_id:
            print(f"[warn] '{qw}' not in vocab, skipping neighbors")
            continue
        nn = top_k_cosine_neighbors(
            qw, vocab=vocab, embeddings=E200, k=5, exclude_self=True)
        print(qw, "->", nn)

    # ---- Re-train for required dimensions using best cfg ----
    required_ds = [50, 100, 200, 300]
    for d in required_ds:
        if d == d_fixed:
            continue
        print(f"\n[final train] d={d} using cfg={cfg}")

        X = Xbest  # reuse built matrix for best window

        W, Wt, b, bt, losses = train_glove_sgd(
            X=X,
            d=d,
            lr=cfg["lr"],
            iters=cfg["iters"],
            x_max=cfg["x_max"],
            alpha=cfg["alpha"],
            seed=1337,
        )
        E = W + Wt
        save_embeddings(OUT_DIR / f"glove_d{d}.npz", vocab, E)


if __name__ == "__main__":
    main()
