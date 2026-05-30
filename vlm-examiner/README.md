# VLM-Examiner — Vision-Language Model MCQ Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Qwen2.5-VL](https://img.shields.io/badge/Model-Qwen2.5--VL--72B-orange.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)

Multi-stage Vision-Language Model pipeline for solving academic multiple-choice questions from images. **95.4% accuracy** on 2000-image synthetic dataset with confidence-based cascading and VRAM-efficient model swapping.

---

## 🎯 System Overview

**Problem:** Given a PNG image containing a deep learning MCQ, output the correct answer (1/2/3/4) or intentionally skip (5).

**Scoring:**
```
final_score = correct - 0.25×wrong - 1×hallucinated
```
- Answer `5` = intentional skip → 0 points (avoids −0.25 penalty)
- Any value outside `{1,2,3,4,5}` = hallucination → −1 point

---

## 🏗️ Pipeline Architecture

```
Image (.png)
  │
  ├─▶ Stage 1: Direct VLM (Qwen2.5-VL-72B, greedy)
  │        confidence ≥ 0.75 → return answer
  │
  ├─▶ Stage 2: Chain-of-Thought VLM (512 tokens reasoning)
  │        confidence ≥ 0.75 → return answer
  │
  ├─▶ [VRAM Swap] Free 72B → load fine-tuned 7B
  │
  ├─▶ Stage 2.5: Fine-Tuned 7B (LoRA-merged)
  │        confidence ≥ 0.80 → return answer
  │
  └─▶ Stage 3: Fallback Solvers (CPU — all GPU freed)
        ├─▶ Math Solver     (SymPy derivatives, algebra)    conf ≥ 0.90
        ├─▶ RAG Retrieval   (FAISS + all-MiniLM-L6-v2)       sim  ≥ 0.80
        └─▶ Heuristic Rules (25 DL-specific regex patterns)
                │
                └─▶ Decision Engine → best answer, or 5 (skip)
```

**Key Innovation:** Confidence-based cascading + VRAM swapping allows 72B → 7B → CPU fallback on 48GB GPU.

---

## 📊 Results

**Offline Evaluation — 2000-image synthetic dataset:**

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

**Per-Stage Breakdown:**
- Stage 1 (Direct VLM): 72.3% solved (confidence ≥ 0.75)
- Stage 2 (CoT VLM): 18.6% solved (total: 90.9%)
- Stage 2.5 (Fine-tuned 7B): 3.1% solved (total: 94.0%)
- Stage 3 (Fallbacks): 1.4% solved, 0.8% skipped

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- **GPU:** NVIDIA L40s or A6000 (48GB VRAM minimum)
- CUDA 11.8+

### Setup

```bash
cd vlm-mcq-solver

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models (72B: ~140GB, 7B: ~14GB)
bash scripts/download_models.sh

# Optional: Download fine-tuned LoRA adapters
huggingface-cli download yourusername/qwen-mcq-lora --local-dir ./models/lora/
```

---

## 🚀 Quick Start

### Single Image Inference

```bash
# Test on single MCQ image
python predict.py --image ./data/sample_mcq.png

# Output:
# 📸 Question: What is the time complexity of...
# 🤖 Predicted Answer: 2 (O(n log n))
# 📊 Confidence: 0.87
# ⏱️ Time: 2.14s
```

### Batch Evaluation

```bash
# Grading script (competition format)
python inference.py --test_dir ./data/test_images/

# Output:
# Processing 2000 images...
# [========================================] 100%
# Final Score: 1889.0 / 2000.0 (94.5%)
```

### API Server

```bash
# Start FastAPI server
python api/server.py --host 0.0.0.0 --port 8000

# In another terminal:
curl -X POST http://localhost:8000/predict \
  -F "image=@sample_mcq.png"

# Response:
# {"answer": 2, "confidence": 0.87, "stage": "stage1_direct"}
```

### Streamlit Dashboard

```bash
# Launch interactive dashboard
streamlit run dashboard.py

# Navigate to http://localhost:8501
# - Upload MCQ images
# - View per-stage accuracy
# - Analyze failure cases
```

---

## 🏗️ Architecture

```
vlm-mcq-solver/
├── src/
│   ├── models/
│   │   ├── qwen_vlm.py           ← Qwen2.5-VL-72B wrapper
│   │   ├── finetuned_vlm.py      ← Fine-tuned 7B wrapper
│   │   └── vram_manager.py       ← Model swapping logic
│   │
│   ├── solvers/
│   │   ├── math_solver.py        ← SymPy symbolic math
│   │   ├── rag_solver.py         ← FAISS + embeddings
│   │   └── heuristic_solver.py   ← Regex patterns
│   │
│   ├── pipeline/
│   │   ├── cascading_pipeline.py ← Main pipeline
│   │   └── confidence_tracker.py ← Per-stage confidence
│   │
│   └── utils/
│       ├── image_preprocessing.py← OCR, normalization
│       ├── prompt_templates.py   ← CoT prompts
│       └── evaluation.py         ← Metrics, scoring
│
├── api/
│   ├── server.py                 ← FastAPI server
│   └── schemas.py                ← Request/response schemas
│
├── scripts/
│   ├── download_models.sh        ← Model downloader
│   ├── finetune_lora.py          ← LoRA fine-tuning script
│   └── generate_synthetic_data.py← MCQ image generator
│
├── notebooks/
│   ├── error_analysis.ipynb      ← Failure case analysis
│   └── ablation_studies.ipynb    ← Stage-wise ablations
│
├── data/
│   ├── synthetic_mcqs/           ← 2000 generated images
│   └── knowledge_base/           ← RAG corpus
│
├── models/
│   ├── qwen2.5-vl-72b/           ← 72B model weights
│   ├── qwen2.5-vl-7b/            ← 7B model weights
│   └── lora/                     ← Fine-tuned adapters
│
├── predict.py                    ← Single-image CLI
├── inference.py                  ← Batch grading script
├── dashboard.py                  ← Streamlit UI
├── requirements.txt              ← Dependencies
└── README.md                     ← This file
```

---

## 🧪 Fine-Tuning (Optional)

```bash
# Prepare training data (1000 MCQ images + answers)
python scripts/prepare_finetuning_data.py --data-dir ./data/train/

# Fine-tune 7B model with LoRA
python scripts/finetune_lora.py \
  --base_model Qwen/Qwen2.5-VL-7B-Instruct \
  --train_data ./data/train/dataset.jsonl \
  --output_dir ./models/lora/qwen-mcq-lora \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4

# Merge LoRA adapters into base model
python scripts/merge_lora.py \
  --base_model ./models/qwen2.5-vl-7b/ \
  --lora_path ./models/lora/qwen-mcq-lora/ \
  --output_path ./models/qwen-7b-mcq-merged/
```

---

## 📝 Key Design Decisions

### 1. Why Multi-Stage Cascading?
- **Stage 1 (Direct):** Fast, solves 72% in 1.5s/image
- **Stage 2 (CoT):** Reasoning helps on ambiguous questions (+18%)
- **Stage 2.5 (Fine-tuned):** Domain-specific knowledge (+3%)
- **Stage 3 (Fallbacks):** Safety net for edge cases (+1.4%)

### 2. Why VRAM Swapping?
- 72B model uses 42GB VRAM (leaves 6GB for KV cache)
- Unloading 72B → Loading 7B takes ~8s (acceptable for 4.5% of questions)
- Alternative: Keep both loaded → requires 80GB VRAM (not available)

### 3. Why Confidence Thresholds?
- **0.75 for VLM:** Empirically optimal (precision/recall trade-off)
- **0.80 for fine-tuned:** Higher bar due to overfitting risk
- **0.90 for math solver:** Symbolic solvers are brittle, require high confidence

**Ablation studies:** [notebooks/ablation_studies.ipynb](./notebooks/ablation_studies.ipynb)

---

## 🤝 Contributing

Priority areas:
- [ ] Support for more VLM backbones (LLaVA, InternVL)
- [ ] Multi-GPU pipeline parallelization
- [ ] Active learning for hard examples
- [ ] Explainability (attention maps, reasoning traces)

---

## 📄 Citation

```bibtex
@misc{sarang2024vlm_examiner,
  author = {Sarang, Yash},
  title = {VLM-Examiner: Multi-Stage Vision-Language MCQ Solver},
  year = {2024},
  url = {https://github.com/YashSarang/deep-learning-research-suite}
}
```

---

## 📚 Related Work

- **Qwen2.5-VL:** [Official Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)
- **Chain-of-Thought:** [Wei et al., 2022](https://arxiv.org/abs/2201.11903)
- **LoRA Fine-Tuning:** [Hu et al., 2021](https://arxiv.org/abs/2106.09685)

---

**⭐ Star if you're building VLM applications!**
