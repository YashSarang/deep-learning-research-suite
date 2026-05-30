"""
Fine-Tuned 7B Fallback Solver — Stage 2.5

Sequential VRAM swap strategy:
  - The 72B model MUST be freed before this module loads the 7B.
  - Call src.vlm.loader.free_memory() in pipeline.py before solve().
  - After solve() returns, free_7b_memory() is called automatically.
  - The 72B model is NOT reloaded after this stage (not needed for
    Stage 3 fallbacks which are all CPU-based).

Robust answer parser: 6-step fallback chain ported from
Current_implementation/evaluate.py (handles JSON, bare letters,
"answer is X", standalone letters, and digits).
"""
from __future__ import annotations

import gc
import json
import re
import torch
from PIL import Image

from src.decision.decision_engine import SolverResult
from src.parser.regex_patterns import LETTER_TO_NUM
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Singleton references for the 7B model ────────────────────────────────────
_model     = None
_processor = None


# ─────────────────────────────────────────────────────────────────────────────
def _load_7b(model_dir: str, use_4bit: bool = True):
    """
    Load the merged fine-tuned 7B model from disk.
    Cached after first call — free with free_7b_memory() when done.

    Args:
        model_dir: Path to the merged (non-LoRA) 7B model directory.
                   Created by scripts/merge_lora.py.
        use_4bit:  Load in NF4 4-bit to minimise VRAM footprint (~3.5 GB).
    """
    global _model, _processor

    if _model is not None:
        return _model, _processor

    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    log.info(f"[FinetunedSolver] Loading 7B model from: {model_dir}")
    _processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    _model.eval()

    mem_alloc = torch.cuda.memory_allocated(0) / 1e9
    log.info(f"[FinetunedSolver] 7B loaded. VRAM allocated: {mem_alloc:.2f} GB")
    return _model, _processor


def free_7b_memory() -> None:
    """Release 7B model from GPU. Call after solve() returns."""
    global _model, _processor
    _model     = None
    _processor = None
    gc.collect()
    torch.cuda.empty_cache()
    log.info("[FinetunedSolver] 7B VRAM freed.")


# ── System prompt (same as finetune training prompt) ─────────────────────────
_SYSTEM_PROMPT = (
    "You are an expert at reading MCQ question papers about concepts related to "
    "Deep learning. Given an image of a question with options, you must select "
    "the correct answer.\n\n"
    "Respond ONLY in this JSON format:\n"
    '{\n'
    '  "question": "<question text>",\n'
    '  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
    '  "answer": "<A|B|C|D>",\n'
    '  "reasoning": "<brief explanation>"\n'
    '}'
)


# ── Robust answer parser (6-step, ported from evaluate.py) ───────────────────
def _parse_robust(raw: str) -> str | None:
    """
    6-step parser that handles every output format the 7B might produce.
    Returns a digit string "1"–"4" or None.
    """
    # Strip leading assistant role token if present
    if "assistant" in raw.lower():
        raw = raw.split("assistant")[-1]
    raw = raw.strip()

    # Step 1 — bare single letter
    if raw.upper() in ("A", "B", "C", "D"):
        return LETTER_TO_NUM.get(raw.upper())

    # Step 2 — full JSON parse
    try:
        letter = json.loads(raw).get("answer", "").strip().upper()
        if letter in LETTER_TO_NUM:
            return LETTER_TO_NUM[letter]
    except Exception:
        pass

    # Step 3 — JSON "answer" key anywhere in text
    m = re.search(r'"answer"\s*:\s*"([ABCD])"', raw, re.IGNORECASE)
    if m:
        return LETTER_TO_NUM.get(m.group(1).upper())

    # Step 4 — natural language "answer is A" / "correct answer: B"
    m = re.search(r'(?:answer|correct)[^\n]*?\b([ABCD])\b', raw, re.IGNORECASE)
    if m:
        return LETTER_TO_NUM.get(m.group(1).upper())

    # Step 5 — any standalone A/B/C/D in text
    m = re.search(r'\b([ABCD])\b', raw.upper())
    if m:
        return LETTER_TO_NUM.get(m.group(1))

    # Step 6 — bare digit 1–4 already in output (shouldn't happen, but safe)
    m = re.search(r'\b([1-4])\b', raw)
    if m:
        return m.group(1)

    return None


# ── Public API ────────────────────────────────────────────────────────────────
def solve(image_path: str, config: dict) -> SolverResult:
    """
    Stage 2.5: Run fine-tuned 7B inference.

    Precondition: The 72B model must already be freed from VRAM
                  (call src.vlm.loader.free_memory() before this).

    The 7B model is loaded, run, and freed within this function.

    Returns:
        SolverResult with answer ∈ {"1","2","3","4"} or None.
        Confidence is 0.82 when a valid answer is parsed (empirically
        calibrated to sit above the finetuned_confidence threshold of 0.80).
    """
    ft_cfg = config.get("finetuned_7b", {})
    if not ft_cfg.get("enabled", True):
        log.info("[FinetunedSolver] Disabled in config — skipping.")
        return SolverResult(answer=None, confidence=0.0)

    model_dir   = config["paths"]["finetuned_model_dir"]
    use_4bit    = (ft_cfg.get("quantization", "4bit") == "4bit")
    max_new_tok = ft_cfg.get("max_new_tokens", 256)

    try:
        model, processor = _load_7b(model_dir, use_4bit)
    except Exception as e:
        log.error(f"[FinetunedSolver] Load failed: {e}")
        return SolverResult(answer=None, confidence=0.0)

    try:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Read this MCQ question and provide the correct answer."},
                ],
            },
        ]

        text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tok,
                do_sample=False,
                use_cache=True,
                temperature=None,
                top_p=None,
            )

        # Strip prompt tokens
        prompt_len = inputs.input_ids.shape[1]
        new_ids    = output_ids[:, prompt_len:]
        raw = processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        log.info(f"[FinetunedSolver] Raw output: {raw!r}")

        ans = _parse_robust(raw)
        # 0.82 — just above finetuned_confidence threshold (0.80)
        # so it will always be trusted when a valid answer is found
        confidence = 0.82 if ans in {"1", "2", "3", "4"} else 0.0

        return SolverResult(answer=ans, confidence=confidence)

    except Exception as e:
        log.error(f"[FinetunedSolver] Inference failed: {e}")
        return SolverResult(answer=None, confidence=0.0)

    finally:
        # Always free 7B VRAM after use — keeps memory clean for Stage 3
        free_7b_memory()
