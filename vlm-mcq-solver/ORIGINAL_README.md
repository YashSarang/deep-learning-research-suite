# Deep Learning Visual MCQ Solver

**GNR638 Competition — Group 2**  
*Yash Sarang (24M2160) · Sarvesh Shashidhar (24M2152) · Anirban Saha (25D1598)*

---

## Overview

An end-to-end Vision-Language Model pipeline that reads PNG images containing deep learning MCQs and outputs the correct answer as `1 / 2 / 3 / 4` (or `5` to intentionally skip).

**Model:** `Qwen2.5-VL-72B-Instruct` (4-bit NF4 quantization)  
**Hardware:** NVIDIA L40s (48 GB VRAM)  
**Entry point (grading):** `inference.py --test_dir <path>`

### Scoring
```
final_score = correct  −  0.25 × wrong  −  1 × hallucinated
```
- `5` = intentional skip → **0 points** (avoids the −0.25 penalty)
- Any value outside `{1,2,3,4,5}` = hallucination → **−1 point**

---

## Pipeline Architecture

```
Image (.png)
  │
  ├─▶ Stage 1: Direct VLM — greedy decode
  │        confidence ≥ 0.75 → return answer
  │
  ├─▶ Stage 2: Chain-of-Thought VLM — extended reasoning window (512 tokens)
  │        confidence ≥ 0.75 → return answer
  │
  ├─▶ [VRAM Swap] Free 72B → load fine-tuned 7B
  │
  ├─▶ Stage 2.5: Fine-Tuned 7B (LoRA-merged)
  │        confidence ≥ 0.80 → return answer
  │
  └─▶ Stage 3: Fallback Solvers (CPU — all GPU freed)
        ├─▶ Math Solver     (SymPy — derivatives, algebra)    conf ≥ 0.90
        ├─▶ RAG Retrieval   (FAISS + all-MiniLM-L6-v2)       sim  ≥ 0.80
        └─▶ Heuristic Rules (25 DL-specific regex patterns)
                │
                └─▶ Decision Engine → best answer, or 5 (skip)
```

---

## Results

**Full offline evaluation — 2000-image synthetic dataset:**

```
============================================================
 OFFLINE EVALUATION RESULTS
============================================================
  Total questions:      2000
  Correct:              1908 (95.4%)
  Wrong:                  76 ( 3.8%)
  Skipped (5):            16 ( 0.8%)
  Hallucinated:            0 ( 0.0%)
  Accuracy:             0.9540
  Competition Score:   1889.0 / 2000.0
  Score Percentage:     94.5%
  Avg time/question:    2.27s
  Total time:           75.6 min
============================================================
```

---

## Repository Structure

```
Kaggle_Competition/
├── inference.py                 ← Grading entry point: --test_dir flag
├── setup.bash                   ← Submission setup script (run once with internet)
├── predict.py                   ← Quick single-image CLI test
├── config.yaml                  ← All paths, thresholds, model settings
├── requirements.txt
│
├── src/
│   ├── pipeline.py              ← Main solve() orchestration
│   ├── vlm/
│   │   ├── loader.py            ← Qwen2.5-VL-72B (NF4 + SDPA)
│   │   ├── inference.py         ← Direct / CoT / extraction paths
│   │   ├── finetuned_solver.py  ← Stage 2.5: LoRA-merged 7B
│   │   └── prompts.py           ← System prompts (1/2/3/4/5 output format)
│   ├── fallback_solvers/
│   │   ├── math_solver.py       ← SymPy symbolic math
│   │   ├── retrieval_solver.py  ← FAISS + MiniLM RAG (offline)
│   │   └── heuristic_solver.py  ← Rule-based DL knowledge
│   ├── decision/
│   │   └── decision_engine.py   ← Confidence-based combiner + skip logic
│   └── retrieval/
│       └── knowledge_base/      ← DL reference text (dl_concepts, formulas, cheatsheets)
│
├── scripts/
│   ├── download_hf_models.py    ← Pull VLM + embedder weights from HuggingFace
│   ├── build_faiss.py           ← Build FAISS index from knowledge base
│   ├── merge_lora.py            ← Merge LoRA adapter → standalone 7B model
│   └── evaluate_offline.py      ← Evaluate on 2000-image synthetic dataset
│
├── data/
│   ├── sample_test_project_2/   ← Official 2-image sample (test format reference)
│   └── synthetic_data/          ← 2000 labelled DL MCQ images for offline eval
│
└── models/                      ← NOT in git — download via setup.bash
    ├── vlm/                     ← Qwen2.5-VL-72B-Instruct weights (~140 GB)
    ├── finetuned_7b/            ← Merged LoRA 7B (built by scripts/merge_lora.py)
    ├── embeddings/              ← all-MiniLM-L6-v2 weights
    └── faiss/                   ← index.faiss + metadata.pkl (built locally)
```

## Submission Setup

### Official grading workflow

```bash
# 1. Run setup (requires internet, run once)
bash setup.bash

# 2. Activate environment
conda activate gnr_project_env

# 3. Run inference
cd GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition
python inference.py --test_dir <absolute_path_to_test_dir>

# 4. Output: submission.csv (in current directory)
```

### Manual setup (cluster / development)

```bash
git clone https://github.com/YashSarang/GNR638-Group2-24M2152-24M2160-25D1598.git
cd GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition

# Create environment
conda create -n gnr_project_env python=3.11 -y
conda activate gnr_project_env

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # optional but recommended

# Download models (requires internet, ~140 GB)
python scripts/download_hf_models.py

# Build FAISS index
python scripts/build_faiss.py
```

### Updating `config.yaml` (if needed)

Default config points to `models/` relative to the project root. To override:

```yaml
paths:
  vlm_model_dir:       "models/vlm"
  embedding_model_dir: "models/embeddings"
  faiss_index:         "models/faiss/index.faiss"
  faiss_metadata:      "models/faiss/metadata.pkl"
```

---

## Running Inference

### Option A — Grading script (recommended)

```bash
python inference.py --test_dir /absolute/path/to/test_data
# Writes: submission.csv  (columns: image_name, option)
```

### Option B — Single image (development / debugging)

```bash
# With real model:
python predict.py data/sample_test_project_2/images/image_1.png

# Mock VLM (test fallback logic only, no GPU needed):
python predict.py data/sample_test_project_2/images/image_1.png --mock
```


---

## Offline Evaluation

Validate the full pipeline on the 2000-image synthetic dataset:

```bash
# Quick check — 50 random questions (~2 min)
python scripts/evaluate_offline.py --n 50 --verbose --save

# Full evaluation — all 2000 images (~75 min)
python scripts/evaluate_offline.py --n 0 --save
```

---

## GPU Optimizations Applied (L40s)

| Optimization | Implementation |
|---|---|
| NF4 double quantization | `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True` |
| BF16 compute dtype | `bnb_4bit_compute_dtype=torch.bfloat16` |
| SDPA attention | `attn_implementation="sdpa"` |
| TF32 matmuls | `torch.backends.cuda.matmul.allow_tf32 = True` |
| Greedy decoding | `do_sample=False` — deterministic, no sampling overhead |
| inference_mode | `torch.inference_mode()` — no autograd tape |
| 95% VRAM budget | `torch.cuda.set_per_process_memory_fraction(0.95)` |
| Single GPU placement | `device_map="cuda:0"` — no CPU offload |

**Expected VRAM:** ~38–42 GB (6–10 GB headroom on L40s)  
**Runtime estimate:** ~2–3 s/question → 2000 questions in **~75 min**

---

## Submission Format

The output `submission.csv` must match this format exactly:

```
image_name,option
image_1,3
image_2,5
image_3,1
...
```

Where:
- `1 / 2 / 3 / 4` = predicted answer option
- `5` = skip (0 points — preferred over a wrong guess at −0.25)

---

## Citations

```bibtex
@article{Qwen2.5-VL,
  title   = {Qwen2.5-VL Technical Report},
  author  = {Qwen Team},
  year    = {2025}
}

@article{johnson2019billion,
  title   = {Billion-scale similarity search with {GPU}s},
  author  = {Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal = {IEEE Transactions on Big Data},
  year    = {2019}
}

@inproceedings{reimers-2019-sentence-bert,
  title     = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {EMNLP 2019},
  year      = {2019}
}
```
