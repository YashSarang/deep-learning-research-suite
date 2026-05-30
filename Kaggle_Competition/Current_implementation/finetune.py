"""
Stage 2: Fine-Tuning with LoRA
CSV columns : id, image_name, option  (option is 1/2/3/4)
Images      : IMAGE_DIR/image_name.png
"""

import os
import torch
import json
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH    = "synthetic_data/synthetic_data/train.csv"
IMAGE_DIR   = "/users/student/pg/pg24/yash.sarang/GNR/GNR638-Group2-24M2152-24M2160-25D1598/Kaggle_Competition/synthetic_data/synthetic_data/images/"
IMAGE_EXT   = ".png"
OUTPUT_DIR  = "./qwen_mcq_finetuned"
TRAIN_SPLIT = 0.9
# ──────────────────────────────────────────────────────────────────────────────

# numeric label → letter the model outputs
OPTION_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}

SYSTEM_PROMPT = """You are an expert at reading MCQ question papers about concepts related to Deep learning. Given an image of a question with options, you must select the correct answer.

Respond ONLY in this JSON format:
{
  "question": "<question text>",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "<A|B|C|D>",
  "reasoning": "<brief explanation>"
}"""


# ── Model ──────────────────────────────────────────────────────────────────────
print("Loading base model ...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()


# ── Dataset ────────────────────────────────────────────────────────────────────
def load_dataset_from_csv(csv_path: str) -> tuple[Dataset, Dataset]:
    """Flat records only — messages are built in collate_fn to avoid PyArrow errors."""
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_SPLIT)

    def to_records(subset):
        records = []
        for _, row in subset.iterrows():
            answer_letter = OPTION_MAP.get(str(row["option"]).strip(), "A")
            img_path = os.path.join(IMAGE_DIR, str(row["image_name"]) + IMAGE_EXT)
            records.append({"image_path": img_path, "answer": answer_letter})
        return records

    return (
        Dataset.from_list(to_records(df[:split_idx])),
        Dataset.from_list(to_records(df[split_idx:]))
    )


# ── Collator with correct label masking ───────────────────────────────────────
def collate_fn(batch):
    """
    Builds chat messages, tokenizes, then masks ALL tokens except the
    assistant response so the model only learns to predict the answer.
    Without this masking, loss stays ~7.6 and the model learns nothing.
    """
    texts, images_list = [], []

    for sample in batch:
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize((512, 512))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Read this MCQ and give the correct answer."},
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps({"answer": sample["answer"]}),
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
        images_list.append([image])

    inputs = processor(
        text=texts,
        images=images_list,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    input_ids = inputs["input_ids"]

    # ── Label masking: only supervise on assistant response tokens ─────────────
    # Start with all -100 (ignored in loss)
    labels = torch.full_like(input_ids, -100)

    # Find the token ids for the assistant separator in Qwen2.5 chat template
    # The template uses: <|im_start|>assistant\n
    assistant_header = "<|im_start|>assistant\n"
    assistant_token_ids = processor.tokenizer.encode(
        assistant_header, add_special_tokens=False
    )
    n = len(assistant_token_ids)

    for i in range(input_ids.shape[0]):
        ids = input_ids[i].tolist()
        found = False
        # Scan for the assistant header token sequence
        for j in range(len(ids) - n):
            if ids[j:j+n] == assistant_token_ids:
                # Supervise on everything AFTER the assistant header
                labels[i, j+n:] = input_ids[i, j+n:]
                found = True
                break

        if not found:
            # Fallback: try plain "\nassistant\n"
            fallback = processor.tokenizer.encode(
                "\nassistant\n", add_special_tokens=False
            )
            nf = len(fallback)
            for j in range(len(ids) - nf):
                if ids[j:j+nf] == fallback:
                    labels[i, j+nf:] = input_ids[i, j+nf:]
                    found = True
                    break

        if not found:
            # Last resort: supervise on last 10 tokens (the answer)
            labels[i, -10:] = input_ids[i, -10:]

    inputs["labels"] = labels
    return inputs


# ── Training args ──────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,      # keep low to avoid OOM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,     # effective batch = 16
    learning_rate=2e-5,                 # 10x lower than before — critical for VLMs
    warmup_steps=100,
    bf16=True,
    save_steps=100,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading dataset ...")
    train_dataset, val_dataset = load_dataset_from_csv(CSV_PATH)
    print(f"  Train : {len(train_dataset)} samples")
    print(f"  Val   : {len(val_dataset)} samples")

    # Verify label masking is working on one sample before full training
    print("\nVerifying label masking on 1 sample ...")
    sample_batch = collate_fn([train_dataset[0]])
    labels = sample_batch["labels"][0]
    n_supervised = (labels != -100).sum().item()
    n_total = labels.shape[0]
    print(f"  Total tokens    : {n_total}")
    print(f"  Supervised tokens: {n_supervised}  (should be small, ~10-50)")
    if n_supervised == 0:
        print("  [ERROR] No supervised tokens found — assistant header not matched!")
        print("  Check the tokenizer output below:")
        print(processor.tokenizer.decode(sample_batch["input_ids"][0]))
    elif n_supervised > 200:
        print("  [WARN] Too many supervised tokens — masking may be wrong")
    else:
        print("  [OK] Label masking looks correct")

    print("\nStarting fine-tuning ...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    trainer.train()

    print(f"\nSaving model -> {OUTPUT_DIR}/final")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    processor.save_pretrained(f"{OUTPUT_DIR}/final")
    print("Done!")