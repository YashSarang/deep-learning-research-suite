# Programming Assignment 1

## Word Embeddings and Named Entity Recognition

This project implements and compares unsupervised word embedding methods (GloVe and SVD) and evaluates them on a supervised Named Entity Recognition (NER) task using CoNLL-2003.  
It also includes a CRF feature-engineering baseline and an optional TF-IDF + SVD improvement.

---

## Project Structure

```text
.
├── data/
│   └── cc_news_filtered.json          # Provided CC-News subset (keys=vocab)
├── outputs/
│   ├── glove_d*.npz                   # GloVe embeddings (Task 1)
│   ├── svd_USigma_d*.npz              # Raw SVD embeddings (Task 2)
│   ├── svd_tfidf_USigma_d*.npz         # TF-IDF SVD embeddings (Task 5)
├── assets/
│   ├── glove_loss_*.png               # GloVe loss curves
│   ├── glove_search_results.csv
│   ├── task4_mlp_results.csv
│   ├── task5_neighbors_raw_vs_tfidf.csv
│   └── task5_tfidf_mlp_metrics.csv
├── data_utils.py                      # Dataset loading utilities
├── utils.py                           # Shared helpers (tokenization, metrics, OOV, etc.)
├── task_1.py                          # Task 1: GloVe training
├── task_2.py                          # Task 2: SVD pipeline
├── task_3.py                          # Task 3: CRF baseline
├── task_4.py                          # Task 4: MLP with embeddings
├── task_5.py                          # Task 5: TF-IDF + SVD (extra credit)
├── requirements.txt
└── README.md
```

## Environment Setup

1. Create a virtual environment (recommended)

   python -m venv venv

   cd venv\Scripts\activate

2. Install dependencies
   pip install -r requirements.txt

3. add data\cc_news_filtered.json (Rename the given 1.1Gb dataset)

4. py task_1.py

### Important:

datasets==3.6.0 is required for CoNLL-2003 (per assignment).

No Jupyter notebooks are used, as required.

## Execution Order (IMPORTANT)

Run the tasks in the following order.

## Step 1 — GloVe training (Task 1)

Tunes hyperparameters with fixed d=200, logs loss curves, then exports embeddings for all required dimensions.

### Run python task_1.py

### Outputs:

outputs/glove_d{d}.npz

assets/glove*loss*\*.png

assets/glove_search_results.csv

## Step 2 — SVD embeddings (Task 2)

Builds the term-document matrix and computes SVD embeddings for the same set of dimensions.

### python task_2.py

### Outputs:

outputs/svd_USigma_d{d}.npz

## Step 3 — CRF baseline (Task 3)

Trains a CRF on CoNLL-2003 using hand-crafted features and reports test performance and feature importance.

### python task_3.py

### Outputs:

Printed test accuracy & macro-F1

Printed list of most important CRF features

## Step 4 — MLP with embeddings (Task 4)

Trains per-token MLP classifiers using GloVe and SVD embeddings at all dimensions.
Handles OOV tokens using a documented strategy (e.g., longest prefix/suffix → <UNK>).

### python task_4.py

### Outputs:

assets/task4_mlp_results.csv

Console summary of best configuration

Comparison with CRF baseline

## Step 5 — TF-IDF + SVD (Extra Credit)

Applies TF-IDF weighting before SVD, compares neighbors, retrains best MLP.

### python task_5.py

### Outputs:

outputs/svd_tfidf_USigma_d{best_d}.npz

assets/task5_neighbors_raw_vs_tfidf.csv

assets/task5_tfidf_mlp_metrics.csv

## Key Implementation Notes

### Vocabulary Constraint

GloVe and SVD training only use the provided vocabulary (~25k tokens).

Tokens outside the vocab are ignored during embedding training.

### OOV Handling (NER Evaluation)

OOV tokens in CoNLL-2003 are handled only at evaluation time.

Strategy used: longest prefix/suffix match → <UNK> fallback.

This strategy is justified in the report.

### Embedding Usage

GloVe: final token vectors = W + W_tilde

SVD: token vectors = U_k Σ_k = X @ V_k

### Metrics

Token-level Accuracy

Macro-averaged F1 score
(Same metrics used across CRF and MLP for fair comparison.)

## Reproducibility

Fixed random seeds are used where applicable.

All intermediate results (loss curves, tables, neighbors) are saved under assets/.

## Submission Checklist

task_1.py … task_5.py (separate files, no notebooks)

utils.py, data_utils.py

requirements.txt

README.md

Report PDF

Assets folder (or Google Drive link if size exceeds limit)
