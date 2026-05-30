"""
Heuristic Solver — deterministic rule-based fallback for common DL concepts.
Maps keyword patterns in questions/options to answer numbers 1/2/3/4.
Low-confidence safety net; only fires when higher-priority solvers fail.
"""
from __future__ import annotations

import re
from src.decision.decision_engine import SolverResult

# ──────────────────────────────────────────────────────────────────────────────
# Rules: (question_pattern, option_keyword_that_should_be_correct)
# These encode well-known DL facts as regex + keyword lookups.
# ──────────────────────────────────────────────────────────────────────────────
RULES: list[tuple[str, str]] = [
    # Overfitting / regularisation
    (r"reduces?\s+variance",               "bagging"),
    (r"reduces?\s+bias",                   "boosting"),
    (r"dropout.*prevents?\s+overfitting",  "regulariz"),
    (r"l2\s+regulariz",                    "weight decay"),
    (r"l1\s+regulariz",                    "sparsity"),

    # Gradient issues
    (r"exploding\s+gradient",              "gradient clipping"),
    (r"vanishing\s+gradient",              "relu"),
    (r"batch\s+normalization",             "internal covariate shift"),

    # Activations
    (r"derivative.*sigmoid",               "sigma.*1.*sigma"),
    (r"relu.*negative",                    "zero"),
    (r"softmax.*output",                   "probability"),

    # Optimisers
    (r"adam\s+optimizer",                  "adaptive"),
    (r"sgd.*momentum",                     "velocity"),

    # Loss functions
    (r"cross[\s-]entropy",                 "log"),
    (r"mean\s+squared\s+error",            "squared"),
    (r"hinge\s+loss",                      "svm"),

    # Architecture
    (r"convolutional.*parameter\s+sharing", "weight sharing"),
    (r"recurrent.*vanishing",               "lstm"),
    (r"attention\s+mechanism",              "query.*key.*value"),
    (r"transformer.*positional",            "sine.*cosine"),
    (r"batch\s+norm.*train",               "mean.*variance"),

    # Definitions
    (r"what\s+is.*deep\s+learning",         "representation"),
    (r"what\s+is.*machine\s+learning",      "data"),
    (r"epoch",                              "one pass through the dataset"),
]

CONFIDENCE = 0.65  # Heuristics are medium-confidence


def solve(extracted_data: dict) -> SolverResult:
    """
    Match question text against known DL rule patterns.
    When a matching option is found, return its number (1/2/3/4).
    """
    ans = SolverResult(answer=None, confidence=0.0)

    question = extracted_data.get("question", "").lower()
    options  = extracted_data.get("options", {})   # {"1": text, ..., "4": text}

    if not question or not options:
        return ans

    for q_pattern, opt_keyword in RULES:
        if not re.search(q_pattern, question, re.IGNORECASE):
            continue
        # Check if any option text contains the expected keyword
        for opt_key, opt_text in options.items():
            if re.search(opt_keyword, opt_text, re.IGNORECASE):
                ans.answer     = str(opt_key)
                ans.confidence = CONFIDENCE
                return ans

    return ans
