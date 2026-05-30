"""
Evaluation Script — compare Stage 1, 2, and 3 results.

Usage:
  # Run inference on fine-tuned model (all data):
  python evaluate.py --run --model_dir ./final --csv synthetic_data/synthetic_data/train.csv --out_csv stage2_results.csv

  # Run on limited samples for quick testing:
  python evaluate.py --run --model_dir ./final --csv synthetic_data/synthetic_data/train.csv --out_csv stage2_results.csv --limit 50

  # Compare stage result CSVs:
  python evaluate.py --stage1 stage1_results.csv --stage2 stage2_results.csv
"""

import argparse
import os
import torch
import json
import re
import pandas as pd
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

# ── Config — edit these ────────────────────────────────────────────────────────
IMAGE_DIR = "/users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition/synthetic_data/synthetic_data/images/"
IMAGE_EXT = ".png"
# ──────────────────────────────────────────────────────────────────────────────

LABEL_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}

SYSTEM_PROMPT = """You are an expert at reading MCQ question papers about concepts related to Deep learning. Given an image of a question with options, you must select the correct answer.

Respond ONLY in this JSON format:
{
  "question": "<question text>",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "<A|B|C|D>",
  "reasoning": "<brief explanation>"
}"""


# ── Answer parser ──────────────────────────────────────────────────────────────
def parse_answer(raw: str) -> str:
    """Robustly extract A/B/C/D from any model output format."""
    # Step 1: extract only the assistant reply
    if "assistant" in raw.lower():
        raw = raw.split("assistant")[-1]
    raw = raw.strip()

    # Step 2: bare single letter
    if raw.upper() in ("A", "B", "C", "D"):
        return raw.upper()

    # Step 3: full JSON
    try:
        return json.loads(raw).get("answer", "ERROR").strip().upper()
    except Exception:
        pass

    # Step 4: JSON pattern in text
    match = re.search(r'"answer"\s*:\s*"([ABCD])"', raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Step 5: "answer is A" / "correct answer: B"
    match = re.search(r'(?:answer|correct)[^\n]*?\b([ABCD])\b', raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Step 6: any standalone letter
    match = re.search(r'\b([ABCD])\b', raw.upper())
    if match:
        return match.group(1)

    return "ERROR"


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame, label: str = "") -> dict:
    total   = len(df)
    correct = df["correct"].sum()
    errors  = (df["predicted"] == "ERROR").sum()

    per_option = (
        df[df["correct"]].groupby("true").size()
        / df.groupby("true").size()
    ).rename("accuracy_per_option")

    print(f"\n{'='*52}")
    if label:
        print(f"  {label}")
    print(f"  Total samples  : {total}")
    print(f"  Correct        : {correct}")
    print(f"  Parse errors   : {errors}")
    print(f"  Accuracy       : {correct/total:.2%}")
    print(f"\n  Per-option accuracy:")
    print(per_option.to_string())
    print(f"{'='*52}")

    return {
        "label":    label,
        "total":    total,
        "correct":  int(correct),
        "errors":   int(errors),
        "accuracy": round(correct / total, 4),
    }


# ── Compare saved result CSVs ──────────────────────────────────────────────────
def compare_stages(*result_csvs, output: str = "eval_report.csv"):
    summary_rows = []
    for label, csv_path in result_csvs:
        if not Path(csv_path).exists():
            print(f"[SKIP] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        metrics = compute_metrics(df, label)
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output, index=False)
    print(f"\nSummary saved -> {output}")
    print(summary_df.to_string(index=False))
    return summary_df


# ── Run inference on fine-tuned model ─────────────────────────────────────────
def run_eval(
    model_dir: str,
    csv_path: str,
    output_csv: str = "eval_finetuned.csv",
    limit: int = None,
):
    print(f"Loading fine-tuned model from {model_dir} ...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir)

    def infer(image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Read this MCQ question and provide the correct answer."},
            ]},
        ]
        text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        raw = processor.decode(output[0], skip_special_tokens=True)
        print(f"[DEBUG] raw = {repr(raw)}")          # ADD THIS LINE
        print(f"[DEBUG] parsed = {parse_answer(raw)}")  # ADD THIS LINE
        return parse_answer(raw)

    df = pd.read_csv(csv_path)

    # ── Limit samples if requested ─────────────────────────────────────────────
    if limit is not None:
        df = df.head(limit)
        print(f"  Running on {limit} samples (--limit {limit})")
    print(f"  Total to evaluate: {len(df)}")

    results, correct = [], 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        img_path    = os.path.join(IMAGE_DIR, str(row["image_name"]) + IMAGE_EXT)
        true_letter = LABEL_TO_LETTER.get(str(row["option"]).strip(), str(row["option"]).strip().upper())

        predicted = "ERROR"
        try:
            predicted = infer(img_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARN] {img_path}: {e}")

        is_correct  = predicted == true_letter
        correct    += int(is_correct)
        results.append({
            "image":     img_path,
            "predicted": predicted,
            "true":      true_letter,
            "correct":   is_correct,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    compute_metrics(results_df, label=f"Fine-tuned: {model_dir}")
    print(f"Per-image results saved -> {output_csv}")
    return results_df


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCQ Pipeline Evaluator")

    # Compare mode
    parser.add_argument("--stage1",    default=None, help="Stage 1 results CSV")
    parser.add_argument("--stage2",    default=None, help="Stage 2 results CSV")
    parser.add_argument("--stage3",    default=None, help="Stage 3 results CSV")
    parser.add_argument("--output",    default="eval_report.csv", help="Summary output CSV")

    # Run mode
    parser.add_argument("--run",       action="store_true", help="Run inference on fine-tuned checkpoint")
    parser.add_argument("--model_dir", default="./final",   help="Path to fine-tuned model")
    parser.add_argument("--csv",       default="synthetic_data/synthetic_data/train.csv", help="Labels CSV")
    parser.add_argument("--out_csv",   default="stage2_results.csv", help="Output results CSV")
    parser.add_argument("--limit",     type=int, default=None, help="Evaluate only first N samples")

    args = parser.parse_args()

    if args.run:
        run_eval(
            model_dir=args.model_dir,
            csv_path=args.csv,
            output_csv=args.out_csv,
            limit=args.limit,
        )
    else:
        stage_pairs = []
        if args.stage1: stage_pairs.append(("Stage 1 — Zero-Shot Baseline", args.stage1))
        if args.stage2: stage_pairs.append(("Stage 2 — Fine-Tuned",         args.stage2))
        if args.stage3: stage_pairs.append(("Stage 3 — Structured Output",  args.stage3))

        if not stage_pairs:
            parser.print_help()
        else:
            compare_stages(*stage_pairs, output=args.output)