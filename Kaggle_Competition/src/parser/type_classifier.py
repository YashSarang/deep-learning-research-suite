"""
Question Type Classifier — determines whether a question is:
  - "mathematical"  → route to math_solver first
  - "computational" → parameter/FLOPs counting, route to math_solver
  - "conceptual"    → route to retrieval/heuristic solver

This runs on the extracted text (not the image directly).
"""
from __future__ import annotations

import re

# ── Pattern sets ──────────────────────────────────────────────────────────────

_MATH_PATTERNS = [
    r"derivative",
    r"gradient",
    r"d/d[wx]",
    r"∂",
    r"integral",
    r"loss\s+function",
    r"backprop",
    r"chain\s+rule",
    r"sigmoid|tanh|relu|softmax",
    r"cross[\s-]entropy",
    r"mean\s+squared",
    r"log[\s(]",
    r"probability\s+of",
]

_COMPUTATIONAL_PATTERNS = [
    r"number\s+of\s+param",
    r"parameters?\s+in",
    r"total\s+param",
    r"flops?",
    r"floating[\s-]point\s+op",
    r"multiply[\s-]add",
    r"compute\s+the",
    r"calculate\s+the",
    r"output\s+size",
    r"receptive\s+field",
    r"memory\s+required",
    r"gflops?",
]

_CONCEPTUAL_PATTERNS = [
    r"what\s+(is|are|does)",
    r"which\s+(of|method|technique|approach)",
    r"why\s+(does|is|would)",
    r"purpose\s+of",
    r"advantage\s+of",
    r"disadvantage",
    r"best\s+describes",
]


def classify(extracted_data: dict) -> str:
    """
    Classify the question type from extracted text.

    Args:
        extracted_data: dict with keys 'question', 'options', 'has_math'.

    Returns:
        "mathematical" | "computational" | "conceptual"
    """
    question  = extracted_data.get("question", "").lower()
    has_math  = extracted_data.get("has_math", False)

    # Explicit flag from extraction
    q_type = extracted_data.get("question_type", "")
    if q_type in ("mathematical", "computational", "conceptual"):
        return q_type

    # Check computational (parameter/FLOPs counting) first
    if any(re.search(p, question, re.IGNORECASE) for p in _COMPUTATIONAL_PATTERNS):
        return "computational"

    # Check purely mathematical
    if has_math or any(re.search(p, question, re.IGNORECASE) for p in _MATH_PATTERNS):
        return "mathematical"

    # Default to conceptual
    return "conceptual"
