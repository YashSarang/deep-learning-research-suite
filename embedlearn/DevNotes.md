# Recommended implementation order (what to build first)

## Step 0 — Project skeleton (do this once)

Create files: task_1.py, task_2.py, task_3.py, task_4.py, optional task_5.py, plus utils.py as shared helpers.

In utils.py, implement:

```bash
tokenizer + normalization rules (consistent across tasks)

vocab lookup + <UNK>/OOV handling hooks

co-occurrence builder (sparse)

nearest-neighbor function (cosine)

metrics helpers (token accuracy, macro-F1)
```

## Step 1 — Data loading & constraints (critical)

Load the provided vocab (~25k) and the filtered CC-News subset and ensure you only train embeddings on this vocab.

Load CoNLL-2003 via datasets==3.6.0 for NER tasks.

## Step 2 — Build the shared co-occurrence machinery (enables Task 1 & 2)

Implement a sliding window co-occurrence counter with window size w (left/right).

Store as scipy.sparse.csr_matrix (or COO then convert) for memory.

This becomes the common “engine”:

GloVe uses co-occurrence counts directly.

SVD uses a matrix representation that you factorize.

## Step 3 — Task 1 (GloVe): train once per window/hparams, then finalize per dimension

Fix d=200 first, tune: window w, learning rate, iterations, and weighting hyperparams (x_max, alpha suggested).

Log loss per iteration; save loss curve plots for different w.

After picking best hyperparams, re-run to export embeddings for each required d.

Deliverables here:

final hyperparams, loss curves, latency notes, and top-5 neighbors for 3 words.

## Step 4 — Task 2 (SVD): compute embeddings for the same d set

Build the matrix (sparse), run TruncatedSVD for each d.

Generate token vectors as their specified product (typically UΣ or similar as described).

Report neighbors for the same 3 words.

## Step 5 — Task 3 (CRF baseline): feature engineering on CoNLL-2003

Create per-token features with context window: lexical, shape, subword/prefix/suffix, etc.

Train CRF on train split, validate on dev, report on test.

In report: list all features and identify most important ones.

## Step 6 — Task 4 (MLP): feature learning with embeddings

For each embedding type (GloVe, SVD) and each dimension d:

Handle OOV tokens (CoNLL tokens not in vocab) with a documented strategy: <UNK >, subword averaging, prefix match, etc.

Train an MLP on (embedding → tag).

Evaluate token accuracy + macro-F1.

Produce the comparison table across all runs + pick the best config; compare with CRF.

## Step 7 — Task 5 (extra credit): TF-IDF + SVD rerun

Apply TF-IDF to the matrix before SVD.

Compare neighbor lists (raw vs TF-IDF) for 5 words, then retrain best MLP and report if it improves.
