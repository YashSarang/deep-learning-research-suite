"""
Constants — shared values referenced across the pipeline.
Import from here rather than hard-coding values in multiple files.
"""
from __future__ import annotations

# ── Answer format ─────────────────────────────────────────────────────────────
VALID_OPTIONS    = {"1", "2", "3", "4"}   # Valid attempted answers
SKIP_OPTION      = "5"                    # Skip — 0 pts, no penalty
ALL_OPTIONS      = VALID_OPTIONS | {SKIP_OPTION}

# Letter → number mapping (safe fallback if VLM outputs A/B/C/D)
LETTER_TO_NUM: dict[str, str] = {
    "A": "1", "B": "2", "C": "3", "D": "4",
    "a": "1", "b": "2", "c": "3", "d": "4",
}

# ── Scoring (from competition rules) ─────────────────────────────────────────
SCORE_CORRECT      =  1.0
SCORE_WRONG        = -0.25
SCORE_SKIP         =  0.0
SCORE_HALLUCINATED = -1.0

# ── Default confidence thresholds ────────────────────────────────────────────
DEFAULT_VLM_CONFIDENCE   = 0.75
DEFAULT_SKIP_THRESHOLD   = 0.55
DEFAULT_MATH_CONFIDENCE  = 0.90
DEFAULT_RAG_SIMILARITY   = 0.80

# ── Model identifiers ─────────────────────────────────────────────────────────
VLM_MODEL_ID   = "Qwen/Qwen2.5-VL-72B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# ── Image processing ──────────────────────────────────────────────────────────
MAX_PIXELS = 1280 * 28 * 28   # Max image resolution for Qwen2.5-VL
MIN_PIXELS =  256 * 28 * 28   # Min image resolution
