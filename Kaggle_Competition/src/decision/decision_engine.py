"""
Decision Engine — combines VLM, fine-tuned 7B, math, retrieval, and heuristic results.

Scoring system (from competition README):
  +1    for correct answer (1/2/3/4)
  -0.25 for incorrect (1/2/3/4 but wrong)
  0     for skip (5)
  -1    for hallucination (anything else)

Strategy: Never hallucinate. Skip (5) when uncertain.

Priority order:
  1. 72B VLM       — high confidence (≥ 0.75)
  2. Fine-tuned 7B — high confidence (≥ 0.80)  ← Stage 2.5
  3. Math solver   — deterministic SymPy (≥ 0.90)
  4. RAG retrieval — FAISS semantic (≥ 0.80)
  5. Heuristic     — rule-based DL patterns
  6. 72B VLM       — medium confidence (≥ 0.55) best-guess
  7. Skip (5)      — 0 points, avoids -0.25 penalty
"""
from __future__ import annotations

VALID_OPTIONS = {"1", "2", "3", "4"}


class SolverResult:
    """Uniform result container for all solver types."""
    __slots__ = ("answer", "confidence")

    def __init__(self, answer: str | None = None, confidence: float = 0.0):
        # answer ∈ {"1","2","3","4"} or None
        self.answer     = answer
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"SolverResult(answer={self.answer!r}, conf={self.confidence:.2f})"


def combine(
    vlm_ans:       SolverResult,
    finetuned_ans: SolverResult,
    math_ans:      SolverResult,
    retrieval_ans: SolverResult,
    heuristic_ans: SolverResult,
    config:        dict,
) -> str:
    """
    Hierarchical decision function.
    Returns a string in {"1","2","3","4","5"}.
    NEVER returns anything outside this set (would be -1 penalty).

    Args:
        vlm_ans:       Result from Stage 1/2 (72B Qwen2.5-VL).
        finetuned_ans: Result from Stage 2.5 (fine-tuned 7B). Pass
                       SolverResult(None, 0.0) if stage was skipped.
        math_ans:      Result from SymPy math solver.
        retrieval_ans: Result from FAISS RAG retrieval.
        heuristic_ans: Result from rule-based heuristics.
        config:        Loaded config.yaml dict.
    """
    t_vlm  = config["thresholds"]["vlm_confidence"]
    t_ft   = config["thresholds"].get("finetuned_confidence", 0.80)
    t_math = config["thresholds"]["math_confidence"]
    t_rag  = config["thresholds"]["retrieval_similarity"]
    t_skip = config["thresholds"]["skip_threshold"]

    print(
        f"[Decision] VLM({vlm_ans.answer}, {vlm_ans.confidence:.2f}) | "
        f"FT7B({finetuned_ans.answer}, {finetuned_ans.confidence:.2f}) | "
        f"Math({math_ans.answer}, {math_ans.confidence:.2f}) | "
        f"RAG({retrieval_ans.answer}, {retrieval_ans.confidence:.2f}) | "
        f"Heuristic({heuristic_ans.answer})"
    )

    # ── 1. 72B VLM with high confidence ─────────────────────────────────
    if vlm_ans.answer in VALID_OPTIONS and vlm_ans.confidence >= t_vlm:
        return vlm_ans.answer

    # ── 2. Fine-tuned 7B with high confidence ────────────────────────────
    if finetuned_ans.answer in VALID_OPTIONS and finetuned_ans.confidence >= t_ft:
        return finetuned_ans.answer

    # ── 3. Math solver (deterministic, trust it highly) ──────────────────
    if math_ans.answer in VALID_OPTIONS and math_ans.confidence >= t_math:
        return math_ans.answer

    # ── 4. RAG retrieval ─────────────────────────────────────────────────
    if retrieval_ans.answer in VALID_OPTIONS and retrieval_ans.confidence >= t_rag:
        return retrieval_ans.answer

    # ── 5. Heuristic ─────────────────────────────────────────────────────
    if heuristic_ans.answer in VALID_OPTIONS:
        return heuristic_ans.answer

    # ── 6. VLM best-guess (medium confidence) ────────────────────────────
    if vlm_ans.answer in VALID_OPTIONS and vlm_ans.confidence >= t_skip:
        return vlm_ans.answer

    # ── 7. Skip — all solvers failed or confidence too low ───────────────
    # Returning 5 = 0 points (no penalty)
    # Returning wrong answer = -0.25 penalty → better to skip
    print("[Decision] All solvers uncertain → skipping (5)")
    return "5"
