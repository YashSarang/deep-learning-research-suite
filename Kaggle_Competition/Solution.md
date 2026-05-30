# Solution Strategy: Qwen2.5-VL-72B Hybrid Pipeline

## Overview

This is a **reasoning pipeline** problem, not a training problem. The approach is a **VLM-first hybrid architecture** that uses the full 48 GB VRAM of the L40s to run the largest capable open-source vision-language model, backed by deterministic fallback solvers.

---

## Results

**Offline evaluation on 2000-image synthetic dataset (full run):**

| Metric | Value |
|---|---|
| Accuracy | **95.4%** (1908 / 2000) |
| Competition Score | **1889.0 / 2000** (94.5%) |
| Wrong answers | 76 (3.8%) |
| Skipped (5) | 16 (0.8%) |
| Hallucinated | 0 (0.0%) |
| Avg time/question | 2.27 s |
| Total time | ~75 min |

---

## Core Strategy

### Primary Engine — Qwen2.5-VL-72B-Instruct (4-bit NF4)

With a 48 GB L40s, we can run **Qwen2.5-VL-72B-Instruct** in NF4 4-bit quantization (~38–42 GB), leaving headroom for context and KV cache. This is SOTA for document and math understanding.

**Why 72B over 7B/32B?**
- 72B has significantly stronger mathematical reasoning
- DL MCQs require understanding LaTeX, derivatives, network architecture calculations
- With 48 GB VRAM, running anything smaller would leave performance on the table

**GPU Optimisations:**
```
NF4 double quantization   → 4-bit weights, minimal accuracy loss
BF16 compute dtype        → Ada Lovelace tensor cores
SDPA attention            → Scaled dot-product attention (stable, no build deps)
TF32 matmuls              → torch.backends.cuda.matmul.allow_tf32 = True
Greedy decoding           → do_sample=False, fully deterministic
torch.inference_mode()    → no autograd tape, max throughput
```

### Inference Pipeline (4-Stage: 72B + 7B + Fallbacks)

```
Stage 1: Direct VLM Answer (greedy, 128 tokens)
  → If confidence ≥ 0.75 → return immediately

Stage 2: Chain-of-Thought VLM (512 tokens, extended reasoning)
  → If confidence ≥ 0.75 → return

[VRAM Swap] Free 72B (gc.collect + cuda.empty_cache) → load fine-tuned 7B

Stage 2.5: Fine-Tuned 7B (LoRA adapter, merged)
  → If confidence ≥ 0.80 → return
  → 7B freed automatically after this stage

Stage 3: Fallback Solvers (CPU only — all GPU freed)
  → Math Solver (SymPy)    → confidence ≥ 0.90 → return
  → RAG Retrieval (FAISS)  → similarity  ≥ 0.80 → return
  → Heuristic Rules        → 25 DL-specific patterns

Stage 4: Decision Engine
  → Best answer above skip threshold (0.55) → return
  → Otherwise → return 5 (skip, 0 points, avoids -0.25 penalty)
```

---

## Answer Format & Scoring

The evaluator uses **numbers 1/2/3/4** (not A/B/C/D):

| Output | Meaning | Score |
|--------|---------|-------|
| 1/2/3/4 (correct) | Answered correctly | +1 |
| 1/2/3/4 (wrong) | Answered incorrectly | −0.25 |
| 5 | Skipped | 0 |
| anything else | Hallucinated | −1 |

**Critical rule:** Never output anything outside {1,2,3,4,5}. Skipping is always better than a random guess (0 > −0.25).

---

## Fallback Solver Details

### Math Solver (SymPy + NumPy)
- For derivatives, integrals, network parameter counting, FLOPs
- Normalises LaTeX → SymPy expressions
- Uses `sympy.simplify()` to test algebraic equivalence between computed result and each option

### RAG Retrieval (FAISS + all-MiniLM-L6-v2)
- Knowledge base: 3 text files (~500 DL concepts, formulas, MCQ patterns)
- Pre-built FAISS flat IP index (cosine similarity on L2-normalized embeddings)
- SentenceTransformer runs on CPU to preserve all GPU VRAM for the VLM

### Heuristic Rules
- 25 regex pattern pairs mapping question keywords → expected option content
- Covers: regularisation, gradients, activations, optimisers, loss functions, architectures
- Lowest priority; fires only when all other solvers are uncertain

---

## Why This Works

| Approach | Result |
|----------|--------|
| Pure OCR + text LLM | Fails on mathematical symbols, LaTeX |
| Small VLM (7B) | Lower reasoning quality on complex DL math |
| **72B VLM + 7B fallback + CPU solvers** | 95.4% accuracy — strong math reasoning + deterministic safety net |
| Training from scratch | Not viable — no training data provided |

The 72B model natively reads PNG images with embedded LaTeX/math notation, eliminating the need for a separate OCR step that would fail on complex formulae. The sequential VRAM swap strategy allows a fine-tuned 7B model to serve as a secondary solver without exceeding the 48 GB VRAM budget.