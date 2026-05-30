"""
Math Solver — uses SymPy and NumPy for deterministic mathematical reasoning.
Returns option number 1/2/3/4 when a math-based answer can be computed.
"""
from __future__ import annotations

import re
import sympy as sp
from src.decision.decision_engine import SolverResult

# ── LaTeX / math pattern normalisation ───────────────────────────────────────
_REPLACEMENTS = [
    (r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)"),
    (r"\\cdot",   "*"),
    (r"\\times",  "*"),
    (r"\\sqrt\{([^}]+)\}", r"sqrt(\1)"),
    (r"\\log",    "log"),
    (r"\\ln",     "ln"),
    (r"\\exp",    "exp"),
    (r"\\sigma",  "sigma"),
    (r"\\tanh",   "tanh"),
    (r"\^",       "**"),
    (r"\$",       ""),
    (r"\\",       ""),
]

def _clean_math(expr: str) -> str:
    for pat, repl in _REPLACEMENTS:
        expr = re.sub(pat, repl, expr)
    return expr.strip()


def _try_simplify_equal(a_str: str, b_str: str) -> bool:
    """Return True if sympy can confirm a == b."""
    try:
        a = sp.sympify(_clean_math(a_str))
        b = sp.sympify(_clean_math(b_str))
        return sp.simplify(a - b) == 0
    except Exception:
        return False


def _evaluate_expression(expr_str: str):
    """Evaluate a math expression with SymPy. Returns symbolic result or None."""
    try:
        return sp.sympify(_clean_math(expr_str))
    except Exception:
        return None


# ── Known closed-form derivative / formula lookup ────────────────────────────
_KNOWN_DERIVATIVES = {
    "sigmoid": "sigma(x)*(1-sigma(x))",
    "tanh":    "1 - tanh(x)**2",
    "relu":    "1 if x>0 else 0",
    "softmax": "p*(1-p)",
}

_KNOWN_FACTS = {
    # Format: (keyword_in_question, answer_keyword_in_option)
    ("cross entropy", "softmax", "derivative"): "-1/p",
    ("mse", "derivative"):                      "2*(y_hat - y)",
}


def solve(extracted_data: dict) -> SolverResult:
    """
    Attempt to solve the MCQ using symbolic mathematics.

    Args:
        extracted_data: dict from VLM extraction with keys:
                        'question', 'options' ({"1":..,"2":..,"3":..,"4":..}),
                        'has_math', 'question_type'.

    Returns:
        SolverResult with answer ∈ {1,2,3,4} or None.
    """
    ans = SolverResult(answer=None, confidence=0.0)

    if not extracted_data.get("has_math"):
        return ans

    question = extracted_data.get("question", "").lower()
    options  = extracted_data.get("options", {})

    if not options:
        return ans

    # ── Pattern 1: derivative of activation function ──────────────────────
    for func_name, derivative_expr in _KNOWN_DERIVATIVES.items():
        if func_name in question and "derivative" in question:
            for opt_key, opt_text in options.items():
                if _try_simplify_equal(opt_text, derivative_expr):
                    ans.answer     = str(opt_key)
                    ans.confidence = 0.92
                    return ans

    # ── Pattern 2: direct numerical evaluation ────────────────────────────
    # Look for "= ?" or "compute X" patterns
    num_match = re.search(r"compute|evaluate|calculate|equals?\s+\?", question)
    if num_match:
        # Try to evaluate each option as a number and cross-check question
        computed = _evaluate_expression(question.split("=")[-1] if "=" in question else "")
        if computed is not None:
            for opt_key, opt_text in options.items():
                opt_val = _evaluate_expression(opt_text)
                if opt_val is not None and sp.simplify(computed - opt_val) == 0:
                    ans.answer     = str(opt_key)
                    ans.confidence = 0.95
                    return ans

    # ── Pattern 3: option-pair simplification ────────────────────────────
    # If two math expressions in options are provably not equal to others
    math_options: dict[str, sp.Expr] = {}
    for k, v in options.items():
        expr = _evaluate_expression(v)
        if expr is not None:
            math_options[k] = expr

    # If only one option can be parsed successfully, it's likely the "clean" answer
    if len(math_options) == 1:
        ans.answer     = list(math_options.keys())[0]
        ans.confidence = 0.5
        return ans

    return ans
