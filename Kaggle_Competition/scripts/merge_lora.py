"""
One-time script: Merge LoRA adapter into a standalone model saved to models/finetuned_7b/.

Run once on the server before using the pipeline:
    python scripts/merge_lora.py

What it does:
  1. Loads Qwen2.5-VL-7B-Instruct base model on CPU (avoids GPU OOM during merge).
  2. Loads the LoRA adapter from Current_implementation/qwen_mcq_finetuned/final/.
  3. Merges LoRA weights into the base model (merge_and_unload()).
  4. Saves the merged model + processor to models/finetuned_7b/.

After this, the pipeline loads from models/finetuned_7b/ directly — no PEFT needed.
"""

import argparse
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BASE  = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_LORA  = os.path.join(
    os.path.dirname(__file__), "..",
    "Current_implementation", "qwen_mcq_finetuned", "final"
)
DEFAULT_OUT   = os.path.join(
    os.path.dirname(__file__), "..", "models", "finetuned_7b"
)


def merge(base_path: str, lora_path: str, out_path: str) -> None:
    os.makedirs(out_path, exist_ok=True)

    print(f"[merge_lora] Base model : {base_path}")
    print(f"[merge_lora] LoRA path  : {lora_path}")
    print(f"[merge_lora] Output dir : {out_path}")

    # Load base on CPU to keep VRAM free (merge does not need GPU)
    print("[merge_lora] Loading base 7B model on CPU...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("[merge_lora] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, lora_path)

    print("[merge_lora] Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"[merge_lora] Saving merged model to {out_path} ...")
    model.save_pretrained(out_path)

    # Save processor (tokenizer + image processor) from LoRA dir
    # (the LoRA dir has tokenizer_config.json, processor_config.json, etc.)
    print("[merge_lora] Saving processor...")
    proc = AutoProcessor.from_pretrained(lora_path)
    proc.save_pretrained(out_path)

    print(f"\n[merge_lora] Done! Merged model saved to: {out_path}")
    print("[merge_lora] Update config.yaml → paths.finetuned_model_dir if needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into standalone model")
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help="Base model path or HF repo (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--lora",
        default=DEFAULT_LORA,
        help="LoRA adapter directory (default: Current_implementation/qwen_mcq_finetuned/final)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output directory for merged model (default: models/finetuned_7b)",
    )
    args = parser.parse_args()
    merge(args.base, args.lora, args.out)
