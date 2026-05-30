"""
Stage 3: Structured Output Enforcement
Loads the fine-tuned LoRA model and forces output to be exactly one of A/B/C/D
using the `outlines` library.
"""

import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
import outlines
from outlines import models, generate

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_DIR       = "./qwen_mcq_finetuned/final"   # output of Stage 2
CSV_PATH       = "labels.csv"                   # for evaluation after stage 3
OUTPUT_CSV     = "stage3_results.csv"
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at reading MCQ question papers about 
concepts related to Deep learning. Given an image of a question with options, 
you must select the single correct answer letter.

Respond with ONLY a single letter: A, B, C, or D"""


# ── Load fine-tuned model ──────────────────────────────────────────────────────
print("Loading base model …")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

print(f"Loading LoRA weights from {LORA_DIR} …")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.merge_and_unload()   # merge LoRA into base weights for outlines compat
model.eval()

processor = AutoProcessor.from_pretrained(LORA_DIR)


# ── Outlines structured generator ─────────────────────────────────────────────
# Wraps the HF model so output is GUARANTEED to be one of ["A","B","C","D"]
outlined_model = models.transformers(model, processor.tokenizer)
generator      = generate.choice(outlined_model, ["A", "B", "C", "D"])


# ── Inference ──────────────────────────────────────────────────────────────────
def build_prompt(image_path: str) -> str:
    """Build the text prompt (image is passed separately to outlines)."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is the correct answer? Reply with only A, B, C, or D."},
            ],
        },
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), image


def run_structured_inference(image_path: str) -> str:
    """Returns guaranteed single letter: A, B, C, or D."""
    prompt, image = build_prompt(image_path)
    # outlines generator enforces the choice constraint at token level
    answer = generator(prompt, images=[image])
    return answer.strip().upper()


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_stage3(csv_path: str, output_csv: str):
    df = pd.read_csv(csv_path)
    results = []
    correct = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stage 3 structured inference"):
        img_path   = row["image_path"]
        true_label = str(row["correct_option"]).strip().upper()

        try:
            predicted = run_structured_inference(img_path)
        except Exception as e:
            predicted = "ERROR"
            print(f"[WARN] {img_path}: {e}")

        is_correct = predicted == true_label
        correct   += int(is_correct)
        results.append({
            "image": img_path,
            "predicted": predicted,
            "true": true_label,
            "correct": is_correct,
        })

    accuracy = correct / len(df)
    print(f"\n=== Stage 3 Accuracy: {accuracy:.2%} ({correct}/{len(df)}) ===")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved → {output_csv}")
    return results_df


if __name__ == "__main__":
    evaluate_stage3(CSV_PATH, OUTPUT_CSV)