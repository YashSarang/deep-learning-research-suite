# Repository Structure

```bash
Kaggle_Competition/
│
├── inference.py                 # ← Grading entry point: python inference.py --test_dir <path>
├── setup.bash                   # ← Submission setup script (run once with internet)
├── predict.py                   # Single-image CLI test (development)
├── config.yaml                  # All paths, thresholds, model settings
├── requirements.txt
│
├── src/
│   ├── pipeline.py              # Main orchestration: solve() + initialize_models()
│   │
│   ├── vlm/
│   │   ├── loader.py            # Qwen2.5-VL-72B loader (NF4, SDPA, BF16)
│   │   ├── inference.py         # Direct + CoT + extraction paths
│   │   ├── finetuned_solver.py  # Stage 2.5 — LoRA-merged 7B with VRAM swap logic
│   │   └── prompts.py           # System prompts (output 1/2/3/4/5)
│   │
│   ├── parser/
│   │   ├── type_classifier.py   # Classify: mathematical / computational / conceptual
│   │   └── regex_patterns.py    # Centralised answer extraction + option parsing
│   │
│   ├── fallback_solvers/
│   │   ├── math_solver.py       # SymPy symbolic math (derivatives, algebra)
│   │   ├── retrieval_solver.py  # FAISS + all-MiniLM-L6-v2 RAG
│   │   └── heuristic_solver.py  # 25 DL keyword rules
│   │
│   ├── decision/
│   │   └── decision_engine.py   # Confidence combiner + skip strategy
│   │
│   ├── retrieval/
│   │   └── knowledge_base/
│   │       ├── dl_concepts.txt  # DL theory reference (~200 concepts)
│   │       ├── formulas.txt     # Mathematical formulas + derivatives
│   │       └── cheatsheets.txt  # MCQ-oriented answer patterns
│   │
│   └── utils/
│       ├── constants.py         # Shared constants (scores, thresholds, model IDs)
│       └── logger.py            # Coloured console logger with timestamps
│
├── scripts/
│   ├── download_hf_models.py    # Pull Qwen2.5-VL-72B + MiniLM from HuggingFace
│   ├── build_faiss.py           # Build FAISS index from knowledge base
│   ├── merge_lora.py            # Merge LoRA adapter → standalone 7B model
│   └── evaluate_offline.py      # Evaluate on 2000-image synthetic dataset
│
├── data/
│   ├── sample_test_project_2/   # Official 2-image sample (format reference)
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   └── images/
│   └── synthetic_data/          # 2000 labelled DL MCQ images + train.csv
│       └── synthetic_data/
│           ├── train.csv
│           └── images/
│
├── models/                      # NOT in git — download via setup.bash
│   ├── vlm/                     # Qwen2.5-VL-72B-Instruct weights (~140 GB)
│   ├── finetuned_7b/            # Merged LoRA 7B (built by scripts/merge_lora.py)
│   ├── embeddings/              # all-MiniLM-L6-v2 weights (~90 MB)
│   └── faiss/                   # Built by scripts/build_faiss.py
│       ├── index.faiss
│       └── metadata.pkl
│
├── gnr_pipeline.sh              # SBATCH: full pipeline (merge + FAISS + eval)
├── gnr_baseline.sh              # SBATCH: zero-shot 72B baseline
└── gnr_finetune.sh              # SBATCH: LoRA fine-tuning of 7B
```

---

## Data Flow

```
inference.py --test_dir <path>
  └─► solve(image_path)  [src/pipeline.py]
        │
        ├─► vlm_engine.solve_direct()         [src/vlm/inference.py]
        │     └─► get_vlm()                   [src/vlm/loader.py]
        │     └─► extract_answer()            [src/parser/regex_patterns.py]
        │
        ├─► vlm_engine.solve_with_cot()       (chain-of-thought, 512 tokens)
        │
        ├─► [VRAM Swap] free_72b() → load 7B
        │
        ├─► finetuned_solver.solve()          [src/vlm/finetuned_solver.py]
        │     └─► _parse_robust()             (6-step answer parser)
        │     └─► free_7b_memory() (auto)
        │
        ├─► vlm_engine.extract_text_and_math()
        │     └─► type_classifier.classify()  [src/parser/type_classifier.py]
        │
        ├─► math_solver.solve()               [src/fallback_solvers/math_solver.py]
        ├─► retrieval_solver.solve()          [src/fallback_solvers/retrieval_solver.py]
        └─► heuristic_solver.solve()          [src/fallback_solvers/heuristic_solver.py]
              │
              └─► decision_engine.combine()   [src/decision/decision_engine.py]
                    └─► Returns "1"/"2"/"3"/"4" or "5" (skip)
```

---

## Critical Design Decisions

| Decision | Rationale |
|----------|-----------|
| Output format: 1/2/3/4/5 | Evaluator expects integers, not A/B/C/D |
| Skip (5) on uncertainty | −0.25 penalty for wrong > 0 for skip |
| Two-stage VLM (direct + CoT) | Direct is fast; CoT handles harder cases |
| Sequential VRAM swap (72B → 7B) | Both models cannot coexist in 48 GB; 7B only loads when 72B fails |
| CPU for embedder | Preserves all 48 GB GPU VRAM for the VLMs |
| FAISS IndexFlatIP | Cosine similarity on L2-normalised vectors — exact, offline |
| NF4 over int8 | Better quality-to-size ratio for large models |
