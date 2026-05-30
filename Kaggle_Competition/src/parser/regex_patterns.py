"""
Regex Patterns — reusable patterns for extracting structured data from VLM output.
Used by the pipeline to robustly parse answers and option text.
"""
from __future__ import annotations

import re

# ── Answer extraction from VLM output ────────────────────────────────────────

# Primary: <answer>X</answer> tags
ANSWER_TAG_PATTERN = re.compile(
    r"<answer>\s*([1-5])\s*</answer>",
    re.IGNORECASE,
)

# Letter variant: <answer>A</answer>
ANSWER_TAG_LETTER_PATTERN = re.compile(
    r"<answer>\s*([A-Da-d])\s*</answer>",
    re.IGNORECASE,
)

# CoT final line: "ANSWER: X" or "Answer: X"
ANSWER_LINE_PATTERN = re.compile(
    r"ANSWER\s*:\s*([1-5])",
    re.IGNORECASE,
)

ANSWER_LINE_LETTER_PATTERN = re.compile(
    r"ANSWER\s*:\s*([A-Da-d])",
    re.IGNORECASE,
)

# Last standalone digit in text (emergency fallback)
STANDALONE_DIGIT_PATTERN = re.compile(r"\b([1-4])\b")

# ── Option text extraction ────────────────────────────────────────────────────

# Matches "1. text" or "1) text" or "(1) text" or "Option 1: text"
OPTION_LINE_PATTERN = re.compile(
    r"(?:option\s*)?([1-4])[.):\s]\s*(.+)",
    re.IGNORECASE,
)

# Matches "A. text" or "A) text" or "(A) text"
OPTION_LETTER_PATTERN = re.compile(
    r"(?:option\s*)?([A-Da-d])[.):\s]\s*(.+)",
    re.IGNORECASE,
)

# ── Math expression patterns ──────────────────────────────────────────────────

LATEX_FRACTION   = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")
LATEX_SQRT       = re.compile(r"\\sqrt\{([^}]+)\}")
LATEX_SUPERSCRIPT = re.compile(r"\^(\{[^}]+\}|[\w])")
INLINE_MATH      = re.compile(r"\$([^$]+)\$")

# ── Utility functions ─────────────────────────────────────────────────────────

LETTER_TO_NUM = {"A": "1", "B": "2", "C": "3", "D": "4",
                 "a": "1", "b": "2", "c": "3", "d": "4"}


def extract_answer(text: str) -> str | None:
    """
    Try all patterns in priority order to extract a digit answer from text.
    Returns "1"–"5" or None.
    """
    # 1. Numeric tag
    m = ANSWER_TAG_PATTERN.search(text)
    if m:
        return m.group(1)

    # 2. Letter tag → convert
    m = ANSWER_TAG_LETTER_PATTERN.search(text)
    if m:
        return LETTER_TO_NUM.get(m.group(1))

    # 3. "ANSWER: N" line
    m = ANSWER_LINE_PATTERN.search(text)
    if m:
        return m.group(1)

    # 4. "ANSWER: A" line → convert
    m = ANSWER_LINE_LETTER_PATTERN.search(text)
    if m:
        return LETTER_TO_NUM.get(m.group(1))

    # 5. Last standalone digit in final 40 chars
    snippet = text[-40:] if len(text) > 40 else text
    digits  = STANDALONE_DIGIT_PATTERN.findall(snippet)
    return digits[-1] if digits else None


def extract_options_from_text(text: str) -> dict[str, str]:
    """
    Parse numbered or lettered options from raw VLM output text.
    Returns {"1": "...", "2": "...", "3": "...", "4": "..."} where possible.
    """
    options: dict[str, str] = {}

    for line in text.splitlines():
        # Try numeric "1. option text"
        m = OPTION_LINE_PATTERN.match(line.strip())
        if m:
            options[m.group(1)] = m.group(2).strip()
            continue

        # Try letter "A. option text"
        m = OPTION_LETTER_PATTERN.match(line.strip())
        if m:
            num = LETTER_TO_NUM.get(m.group(1).upper())
            if num:
                options[num] = m.group(2).strip()

    return options
