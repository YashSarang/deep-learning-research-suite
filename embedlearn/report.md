# Programming Assignment 1 Report

_Generated draft: 2026-02-14 03:12_

---

## 1. Executive Summary

This report compares unsupervised word embedding methods (GloVe vs. SVD) trained on a provided CC-News subset and evaluates their effectiveness on Named Entity Recognition (NER) using CoNLL-2003. We also compare against a CRF baseline using hand-crafted lexical/shape/subword features.

Best MLP configuration (by test Macro-F1): **GLOVE**, **d=300**, test Macro-F1=0.3937023592717927, test Acc=0.8617206848282546.


---

## 2. Data & Evaluation Protocol

- **Unsupervised training data:** Provided CC-News subset (~67k documents) and provided vocabulary (~25k tokens).

- **Constraint:** GloVe and SVD training performed only on provided vocabulary (OOV ignored during training).

- **NER data:** CoNLL-2003 (train/validation/test splits as provided).

- **Metrics:** Token-level Accuracy and Macro-F1; final metrics reported on **test** split.


---

## 3. Task 1 — GloVe Pre-training

### 3.1 Objective and Setup

GloVe is trained using a weighted least-squares objective on a global word–word co-occurrence matrix built with a context window `w`.

### 3.2 Hyperparameter Search (fixed d=200)

Search results:


| w | lr | iters | x_max | alpha | final_loss | total_train_s | nnz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.025 | 30 | 100.0 | 0.75 | 0.03230632371989255 | 1936.5457620620728 | 7249560 |
| 2 | 0.01 | 30 | 100.0 | 0.75 | 0.03413386779517426 | 1931.5652570724487 | 7249560 |
| 2 | 0.005 | 30 | 100.0 | 0.75 | 0.046281528433162585 | 1938.0983037948608 | 7249560 |
| 5 | 0.025 | 30 | 100.0 | 0.75 | 0.03880991018813584 | 3986.3844265937805 | 14991490 |
| 5 | 0.01 | 30 | 100.0 | 0.75 | 0.03692408292896721 | 3990.2024261951447 | 14991490 |
| 5 | 0.005 | 30 | 100.0 | 0.75 | 0.041410241958888774 | 3983.4831500053406 | 14991490 |
| 10 | 0.025 | 30 | 100.0 | 0.75 | 0.03969762908673617 | 6057.036014556885 | 22583676 |
| 10 | 0.01 | 30 | 100.0 | 0.75 | 0.037169590137854695 | 6071.296098232269 | 22583676 |
| 10 | 0.005 | 30 | 100.0 | 0.75 | 0.03936383268943316 | 6891.406728029251 | 22583676 |



### 3.3 Loss Curves

![GloVe Loss Curve](assets\glove_loss_w10_d200_lr0.005.png)

![GloVe Loss Curve](assets\glove_loss_w10_d200_lr0.01.png)

![GloVe Loss Curve](assets\glove_loss_w10_d200_lr0.025.png)

![GloVe Loss Curve](assets\glove_loss_w2_d200_lr0.005.png)

![GloVe Loss Curve](assets\glove_loss_w2_d200_lr0.01.png)

![GloVe Loss Curve](assets\glove_loss_w2_d200_lr0.025.png)

![GloVe Loss Curve](assets\glove_loss_w2_d200_lr0.05.png)

![GloVe Loss Curve](assets\glove_loss_w5_d200_lr0.005.png)

![GloVe Loss Curve](assets\glove_loss_w5_d200_lr0.01.png)

![GloVe Loss Curve](assets\glove_loss_w5_d200_lr0.025.png)

![GloVe Loss Curve](assets\glove_loss_w5_d200_lr0.05.png)


### 3.4 Nearest Neighbors (Top-5)

Words used (must match Task 2): **united, city, president**


_Paste your printed neighbor lists from task_1.py here (Top-5 for each word)._ 


---

## 4. Task 2 — SVD Pipeline

SVD is applied to a sparse term-document matrix X. Token representations are computed as $U_k\Sigma_k$.


### 4.1 Nearest Neighbors (Top-5)

Same 3 words as Task 1: **united, city, president**


_Paste your printed neighbor lists from task_2.py here (Top-5 for each word)._ 


---

## 5. Task 3 — CRF Baseline (Feature Engineering)

### 5.1 Feature Set

_List every feature included in the CRF here (copy from your implementation)._ 


### 5.2 Test Results and Feature Importance

_Paste test Accuracy, Macro-F1, and top weighted CRF features here (from task_3.py output)._ 


---

## 6. Task 4 — MLP (Feature Learning with Embeddings)

### 6.1 Architecture

MLP takes a single token embedding as input and predicts one of 9 NER tags.


### 6.2 OOV Strategy

_Describe and justify your OOV strategy (e.g., longest prefix/suffix match → <UNK> fallback)._ 


### 6.3 Results Table (All Runs)

| algo | d | oov_mode | test_acc | test_macro_f1 | best_dev_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| glove | 50 | affix | 0.829762032949284 | 0.13005777932834406 | 0.1370765840563632 |
| glove | 100 | affix | 0.833616883816087 | 0.15317088844312904 | 0.1641526359096762 |
| glove | 200 | affix | 0.8496823516743836 | 0.2737348763782632 | 0.30005442394357007 |
| glove | 300 | affix | 0.8617206848282546 | 0.3937023592717927 | 0.41428247802986434 |
| svd | 50 | affix | 0.8253472596102078 | 0.10086394219306578 | 0.10380914859720554 |
| svd | 100 | affix | 0.8253687950899107 | 0.10218609921314135 | 0.10498975150492013 |
| svd | 200 | affix | 0.8251965112522881 | 0.10310935439002292 | 0.10733608897190824 |
| svd | 300 | affix | 0.8253472596102078 | 0.10320772534351698 | 0.10730401210399793 |



### 6.4 Best Configuration and Comparison vs CRF

Best MLP: **GLOVE**, d=300, test Macro-F1=0.3937023592717927, test Acc=0.8617206848282546.

_Compare the best MLP vs CRF and explain why one outperformed the other._


---

## 7. Task 5 — Extra Credit (TF-IDF + SVD)

### 7.1 Quality Check 1: Neighbors (Raw vs TF-IDF)

Words: **city, run, music, london, market**


_See CSV: `assets\task5_neighbors_raw_vs_tfidf.csv`_


### 7.2 Quality Check 2: MLP using TF-IDF SVD vectors

_See CSV: `assets\task5_tfidf_mlp_metrics.csv`_



---

## 8. GenAI Usage Disclosure

This project used an LLM assistant during implementation. Per course policy, the usage is disclosed here with shareable links.


_Add shareable chat links here before submitting._


