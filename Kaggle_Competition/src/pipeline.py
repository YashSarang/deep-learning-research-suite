"""
Main Pipeline — orchestrates VLM + fine-tuned fallback + additional solvers.

Answer format: integer string in {"1","2","3","4","5"}
  1/2/3/4 = chosen option
  5       = intentional skip (avoids -0.25 penalty)

Flow:
  1. Direct VLM shot (72B) → if confidence ≥ t_vlm: return
  2. Chain-of-thought VLM (72B) → if confidence ≥ t_vlm: return
  ── VRAM swap: free 72B ──────────────────────────────────────
  2.5. Fine-tuned 7B (LoRA-merged) → if confidence ≥ t_ft: return
  ── 7B auto-freed inside finetuned_solver.solve() ────────────
  3. Extract structured data (no model needed — uses cached CoT text)
     → run fallback solvers (Math/RAG/Heuristic) — all CPU
  4. Decision engine combines all signals
  5. Skip if all signals are weak
"""
from __future__ import annotations

import os
import yaml

from src.vlm import inference as vlm_engine
from src.fallback_solvers import math_solver, retrieval_solver, heuristic_solver
from src.decision import decision_engine

# ── Singleton config ──────────────────────────────────────────────────────────
_config: dict | None = None


def _resolve_path(relative: str) -> str:
    """
    Resolve path: try local first, then scan /kaggle/input/ datasets.
    """
    if os.path.exists(relative):
        return relative

    kaggle_base = "/kaggle/input"
    if os.path.isdir(kaggle_base):
        for ds_dir in sorted(os.listdir(kaggle_base)):
            candidate = os.path.join(kaggle_base, ds_dir, relative)
            if os.path.exists(candidate):
                return candidate

    return relative  # Return as-is; will fail with a clear error later


def load_config(config_path: str = "config.yaml") -> dict:
    """Load and resolve config.yaml. Caches result."""
    global _config
    if _config is not None:
        return _config

    resolved = _resolve_path(config_path)
    with open(resolved, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve all path values
    if "paths" in cfg:
        cfg["paths"] = {k: _resolve_path(v) for k, v in cfg["paths"].items()}

    _config = cfg
    return _config


def initialize_models(config_path: str = "config.yaml") -> None:
    """
    Called ONCE at notebook startup.
    Loads the 72B VLM into GPU memory. Subsequent calls are no-ops (cached).
    The fine-tuned 7B is NOT pre-loaded here — it uses lazy sequential loading.
    """
    cfg = load_config(config_path)
    vlm_dir = cfg["paths"]["vlm_model_dir"]
    use_4bit = (cfg["vlm"].get("quantization") == "4bit")
    print(f"[Pipeline] Initialising VLM: {cfg['vlm']['model_id']} | 4-bit={use_4bit}")
    from src.vlm.loader import get_vlm
    get_vlm(vlm_dir, use_4bit=use_4bit)
    print("[Pipeline] 72B VLM ready. Fine-tuned 7B will load on-demand (sequential swap).")


def solve(image_path: str, config_path: str = "config.yaml") -> str:
    """
    Solve one MCQ image.

    Args:
        image_path:  Path to the PNG image.
        config_path: Path to config.yaml.

    Returns:
        str in {"1","2","3","4","5"}.
        "5" = intentional skip.

    VRAM Timeline:
        [Stage 1-2]  72B loaded  → Direct + CoT inference
        [swap]       72B freed   → torch.cuda.empty_cache()
        [Stage 2.5]  7B loaded   → Finetuned inference → 7B freed (auto)
        [Stage 3-4]  CPU only    → Math / RAG / Heuristic / Decision
    """
    cfg    = load_config(config_path)
    t_vlm  = cfg["thresholds"]["vlm_confidence"]
    t_ft   = cfg["thresholds"].get("finetuned_confidence", 0.80)
    t_skip = cfg["thresholds"]["skip_threshold"]

    # ── Stage 1: Direct VLM (greedy, low tokens) ──────────────────────────
    print(f"\n[Pipeline] === Solving: {os.path.basename(image_path)} ===")
    vlm_result = vlm_engine.solve_direct(image_path, cfg)

    if vlm_result.answer in {"1","2","3","4"} and vlm_result.confidence >= t_vlm:
        print(f"[Pipeline] Stage 1 answer: {vlm_result.answer} (conf={vlm_result.confidence:.2f})")
        return vlm_result.answer

    # ── Stage 2: Chain-of-thought VLM (more reasoning tokens) ─────────────
    print(f"[Pipeline] Stage 1 conf={vlm_result.confidence:.2f} → trying CoT (Stage 2)...")
    cot_result = vlm_engine.solve_with_cot(image_path, cfg)

    if cot_result.answer in {"1","2","3","4"} and cot_result.confidence >= t_vlm:
        print(f"[Pipeline] Stage 2 CoT answer: {cot_result.answer} (conf={cot_result.confidence:.2f})")
        return cot_result.answer

    # Best 72B result so far (used later in decision engine)
    best_vlm = vlm_result if vlm_result.confidence >= cot_result.confidence else cot_result
    best_vlm_ans = decision_engine.SolverResult(
        answer=best_vlm.answer, confidence=best_vlm.confidence
    )

    # ── VRAM Swap: free 72B before loading 7B ─────────────────────────────
    print(f"[Pipeline] 72B uncertain (conf={best_vlm.confidence:.2f}) → swapping to 7B...")
    from src.vlm.loader import free_memory as free_72b
    free_72b()
    # free_memory() already calls gc.collect() + cuda.empty_cache()

    # ── Stage 2.5: Fine-tuned 7B (loads, infers, and frees itself) ─────────
    from src.vlm.finetuned_solver import solve as ft_solve
    ft_result = ft_solve(image_path, cfg)
    # Note: finetuned_solver.solve() calls free_7b_memory() in its finally block

    if ft_result.answer in {"1","2","3","4"} and ft_result.confidence >= t_ft:
        print(f"[Pipeline] Stage 2.5 fine-tuned 7B answer: {ft_result.answer} "
              f"(conf={ft_result.confidence:.2f})")
        return ft_result.answer

    print(f"[Pipeline] Stage 2.5 uncertain (conf={ft_result.confidence:.2f}) "
          f"→ running CPU fallbacks (Stage 3)...")

    # ── Stage 3: Extract structured data → fallback solvers ───────────────
    # Re-use the cached CoT raw text for extraction — no model reload needed.
    # If extraction fails, math/RAG/heuristic still run (they handle empty data).
    extracted = vlm_engine.extract_text_and_math(image_path, cfg)

    math_result      = math_solver.solve(extracted)
    retrieval_result = retrieval_solver.solve(extracted, cfg)
    heuristic_result = heuristic_solver.solve(extracted)

    # ── Stage 4: Decision engine ───────────────────────────────────────────
    final = decision_engine.combine(
        vlm_ans       = best_vlm_ans,
        finetuned_ans = ft_result,
        math_ans      = math_result,
        retrieval_ans = retrieval_result,
        heuristic_ans = heuristic_result,
        config        = cfg,
    )

    print(f"[Pipeline] Final answer: {final}")
    return final
