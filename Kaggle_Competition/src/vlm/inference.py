"""
VLM Inference — Qwen2.5-VL-72B on L40s
Outputs answer as integer 1/2/3/4 or 5 (skip).
Uses torch.inference_mode() for maximum GPU throughput.
"""
from __future__ import annotations

import json
import re
import torch
from PIL import Image

from src.vlm.loader import get_vlm
from src.vlm.prompts import (
    DIRECT_ANSWER_PROMPT,
    EXTRACTION_PROMPT,
    CHAIN_OF_THOUGHT_PROMPT,
)
from src.parser.regex_patterns import extract_answer, extract_options_from_text, LETTER_TO_NUM
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Valid answer mapping ──────────────────────────────────────────────────────
# LETTER_TO_NUM and extract_answer() imported from src.parser.regex_patterns

# ── CoT text cache ────────────────────────────────────────────────────────────
# Stores the last solve_with_cot() raw output so extract_text_and_math() can
# parse structured data from it WITHOUT reloading the 72B model after the
# VRAM swap to Stage 2.5.
_last_cot_raw: str = ""


class VLMResult:
    """Container for a VLM prediction."""
    __slots__ = ("answer", "confidence", "raw_text")

    def __init__(self, answer: str | None = None, confidence: float = 0.0, raw_text: str = ""):
        # answer ∈ {"1","2","3","4","5",None}
        self.answer     = answer
        self.confidence = confidence
        self.raw_text   = raw_text

    def __repr__(self) -> str:
        return f"VLMResult(answer={self.answer!r}, confidence={self.confidence:.2f})"


# ──────────────────────────────────────────────────────────────────────────────
def _build_messages(image: Image.Image, prompt: str) -> list[dict]:
    """Build the chat message list for Qwen2.5-VL."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]


def _generate(model_path: str, image_path: str, prompt: str, max_new_tokens: int = 128) -> str:
    """
    Core generation call — all GPU operations inside torch.inference_mode().
    Returns decoded output string.
    """
    from qwen_vl_utils import process_vision_info  # Qwen2.5-VL utility

    processor, model = get_vlm(model_path)
    image = Image.open(image_path).convert("RGB")
    messages = _build_messages(image, prompt)

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Extract vision inputs
    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenize
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    # Generate — greedy (temperature=0 equivalent)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # Greedy decoding — deterministic
            use_cache=True,           # KV cache enabled
            temperature=None,        # Disable temperature (greedy)
            top_p=None,              # Disable top-p (greedy)
        )

    # Strip prompt tokens from output
    prompt_len = inputs.input_ids.shape[1]
    new_ids    = generated_ids[:, prompt_len:]

    output = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # Free intermediate tensors immediately
    del inputs, generated_ids, new_ids
    torch.cuda.empty_cache()

    return output.strip()


# ── Answer parsing helpers ────────────────────────────────────────────────────

def _parse_answer_tag(text: str) -> str | None:
    """Extract digit from <answer>X</answer> tag."""
    m = re.search(r"<answer>\s*([1-5])\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: letter in tag
    m = re.search(r"<answer>\s*([A-Da-d])\s*</answer>", text, re.IGNORECASE)
    if m:
        return LETTER_TO_NUM.get(m.group(1).upper())
    return None


def _parse_answer_line(text: str) -> str | None:
    """Extract digit from 'ANSWER: X' pattern (chain-of-thought output)."""
    m = re.search(r"ANSWER\s*:\s*([1-5])", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Fallback: ANSWER: A/B/C/D
    m = re.search(r"ANSWER\s*:\s*([A-Da-d])", text, re.IGNORECASE)
    if m:
        return LETTER_TO_NUM.get(m.group(1).upper())
    return None


def _extract_standalone_digit(text: str) -> str | None:
    """
    Last resort: find a standalone 1-4 in the last 20 chars.
    Explicitly excludes 5 here since standalone '5' could be noise.
    """
    snippet = text[-40:] if len(text) > 40 else text
    digits  = re.findall(r"\b([1-4])\b", snippet)
    return digits[-1] if digits else None


# ── Public inference APIs ─────────────────────────────────────────────────────

def solve_direct(image_path: str, config: dict) -> VLMResult:
    """
    Primary path: ask VLM to output answer directly.
    Returns VLMResult with answer ∈ {"1","2","3","4","5"} and a confidence score.
    """
    model_path = config["paths"]["vlm_model_dir"]
    max_tok    = config["vlm"].get("max_new_tokens", 128)

    raw = _generate(model_path, image_path, DIRECT_ANSWER_PROMPT, max_tok)
    print(f"[VLM Direct] {raw!r}")

    ans = _parse_answer_tag(raw)
    confidence = 0.9 if ans in {"1", "2", "3", "4"} else 0.0

    # Handle explicit skip from model
    if ans == "5":
        return VLMResult(answer="5", confidence=0.0, raw_text=raw)

    if ans is None:
        # Standalone digit fallback
        ans = _extract_standalone_digit(raw)
        confidence = 0.5 if ans else 0.0

    return VLMResult(answer=ans, confidence=confidence, raw_text=raw)


def solve_with_cot(image_path: str, config: dict) -> VLMResult:
    """
    Secondary path: chain-of-thought reasoning.
    Used when direct answer confidence is in the 'uncertain' zone.
    Allows more tokens for full reasoning.
    Caches raw output in _last_cot_raw so extract_text_and_math() can
    reuse it without reloading the model after the VRAM swap.
    """
    global _last_cot_raw
    model_path = config["paths"]["vlm_model_dir"]

    raw = _generate(model_path, image_path, CHAIN_OF_THOUGHT_PROMPT, max_new_tokens=512)
    _last_cot_raw = raw  # Cache for extract_text_and_math()
    log.info(f"[VLM CoT] {raw!r}")

    ans = extract_answer(raw)
    confidence = 0.85 if ans in {"1", "2", "3", "4"} else 0.0
    if ans == "5":
        confidence = 0.0

    return VLMResult(answer=ans, confidence=confidence, raw_text=raw)


def _parse_extraction_json(raw: str) -> dict | None:
    """Try to parse structured extraction JSON from raw text. Returns None on failure."""
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        data = json.loads(cleaned)
        opts = data.get("options", {})
        if not opts:
            return None
        normalised: dict[str, str] = {}
        for k, v in opts.items():
            norm_key = LETTER_TO_NUM.get(str(k).upper(), str(k))
            normalised[norm_key] = str(v)
        data["options"] = normalised
        return data
    except Exception:
        return None


def extract_text_and_math(image_path: str, config: dict) -> dict:
    """
    Fallback path: extract structured JSON (question + 4 options) from image.
    Used when VLM confidence is too low even after CoT.

    Strategy (avoids reloading the 72B after VRAM swap):
      1. Try to parse question/options from the cached CoT raw text.
      2. Only if that fails, reload the 72B and run a dedicated extraction prompt.

    Returns dict with keys: question, options (1-4), has_math, question_type.
    """
    _empty = {"question": "", "options": {}, "has_math": False, "question_type": "conceptual"}

    # ── Attempt 1: parse from cached CoT text (no model reload needed) ───────
    if _last_cot_raw:
        parsed = _parse_extraction_json(_last_cot_raw)
        if parsed and parsed.get("options"):
            print("[VLM Extract] Parsed from cached CoT output — no model reload needed.")
            return parsed

    # ── Attempt 2: reload 72B and run dedicated extraction prompt ────────────
    # This path is only hit if the CoT text had no parseable structure.
    print("[VLM Extract] CoT cache insufficient — reloading 72B for extraction...")
    model_path = config["paths"]["vlm_model_dir"]
    try:
        raw = _generate(model_path, image_path, EXTRACTION_PROMPT, max_new_tokens=512)
        print(f"[VLM Extract] {raw!r}")
        parsed = _parse_extraction_json(raw)
        return parsed if parsed else _empty
    except Exception as e:
        print(f"[VLM Extract] Generation failed: {e}")
        return _empty
