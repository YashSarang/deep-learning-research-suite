"""
Stage 1: Baseline (Zero-Shot, No Training)
Run this BEFORE fine-tuning to establish a baseline accuracy.
"""

import os
import torch
import json
import re
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm

# ── Model ──────────────────────────────────────────────────────────────────────
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert at reading MCQ question papers about 
concepts related to Deep learning. Given an image of a question with options, 
you must:
1. Extract the question text
2. Extract all options (A, B, C, D)
3. Select the correct answer

Respond ONLY in this JSON format:
{
  "question": "<question text>",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "<A|B|C|D>",
  "reasoning": "<brief explanation>"
}"""


def run_inference(image_path: str) -> dict:
    """Run zero-shot inference on a single image. Returns parsed JSON dict."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Read this MCQ question and provide the correct answer."},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)

    response = processor.decode(output[0], skip_special_tokens=True)
    raw = response.split("assistant")[-1].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON block if model added extra text
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"answer": "ERROR", "raw": raw}


def run_baseline(csv_path: str, image_dir: str, output_csv: str = "baseline_results.csv"):
    """
    Args:
        csv_path   : Path to CSV with columns [image_name, option]
        image_dir  : Root directory containing the images
        output_csv : Where to save per-image results
    """
    # Maps model output (A/B/C/D) → your CSV labels (1/2/3/4)
    OPTION_MAP = {"A": "1", "B": "2", "C": "3", "D": "4"}

    df = pd.read_csv(csv_path)
    results = []
    correct = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Zero-shot baseline"):
        img_path  = os.path.join(image_dir, row["image_name"] + ".png")
        true_label = str(row["option"]).strip()   # "1", "2", "3", or "4"

        try:
            pred = run_inference(img_path)
            predicted_letter = str(pred.get("answer", "ERROR")).strip().upper()
            predicted = OPTION_MAP.get(predicted_letter, "ERROR")
        except Exception as e:
            predicted = "ERROR"
            print(f"[WARN] {img_path}: {e}")

        is_correct = predicted == true_label
        correct += int(is_correct)
        results.append({
            "image": img_path,
            "predicted": predicted,
            "true": true_label,
            "correct": is_correct,
        })

    accuracy = correct / len(df)
    print(f"\n=== Stage 1 Baseline Accuracy: {accuracy:.2%} ({correct}/{len(df)}) ===")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved → {output_csv}")
    return results_df

if __name__ == "__main__":
    # ── Edit these paths ───────────────────────────────────────────────────────
    CSV_PATH  = "/users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition/synthetic_data/synthetic_data/train.csv"          # columns: image_path, correct_option
    IMAGE_DIR = "/users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition/synthetic_data/synthetic_data/images/"             # root dir (used only for reference)
    OUTPUT    = "/users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition/stage1_results.csv"
    # ──────────────────────────────────────────────────────────────────────────
    run_baseline(CSV_PATH, IMAGE_DIR, OUTPUT)