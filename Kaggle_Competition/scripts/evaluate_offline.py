"""
Offline Evaluator — Tests the full pipeline on the synthetic dataset.

Computes:
  - Accuracy
  - Competition score = correct - 0.25 * wrong - 1.0 * hallucinated
  - Skip rate, per-option breakdown

Usage:
    python scripts/evaluate_offline.py
    python scripts/evaluate_offline.py --n 50 --verbose
    python scripts/evaluate_offline.py --config config.yaml --n 100
"""
import os
import sys
import argparse
import time
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALID_ANSWERS    = {"1", "2", "3", "4", "5"}
COMPETITION_OPTS = {"1", "2", "3", "4"}


def score_prediction(pred: str, truth: str) -> float:
    """Return competition score for a single prediction."""
    if pred not in VALID_ANSWERS:
        return -1.0    # Hallucination
    if pred == "5":
        return 0.0     # Skip
    if pred == str(truth):
        return 1.0     # Correct
    return -0.25       # Wrong


def evaluate(args):
    from src.pipeline import solve, load_config, initialize_models

    cfg = load_config(args.config)
    paths = cfg["paths"]

    csv_path    = paths.get("synthetic_data_csv", args.csv)
    images_dir  = paths.get("synthetic_data_images", args.images_dir)

    print(f"[Eval] CSV: {csv_path}")
    print(f"[Eval] Images: {images_dir}")

    df = pd.read_csv(csv_path)
    # Expected columns: id, image_name, option
    if "option" not in df.columns:
        raise ValueError("CSV must have 'option' column with ground truth labels")

    if args.n > 0:
        df = df.sample(n=min(args.n, len(df)), random_state=42).reset_index(drop=True)
        print(f"[Eval] Sampling {len(df)} questions")
    else:
        print(f"[Eval] Evaluating all {len(df)} questions")

    print("\n[Eval] Initialising models (this may take several minutes)...")
    initialize_models(args.config)

    results = []
    total_time = 0.0
    per_class_counts = defaultdict(lambda: {"correct": 0, "wrong": 0, "skip": 0, "hall": 0})

    print("\n[Eval] Starting evaluation...\n")
    for i, row in df.iterrows():
        img_name = row["image_name"]
        truth    = str(row["option"])
        img_path = os.path.join(images_dir, f"{img_name}.png")

        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        t0   = time.time()
        pred = solve(img_path, config_path=args.config)
        dt   = time.time() - t0
        total_time += dt

        sc   = score_prediction(pred, truth)
        results.append({"image": img_name, "truth": truth, "pred": pred, "score": sc, "time": dt})

        # Per-class breakdown
        cls = str(truth)
        if pred == truth:
            per_class_counts[cls]["correct"] += 1
        elif pred == "5":
            per_class_counts[cls]["skip"] += 1
        elif pred not in VALID_ANSWERS:
            per_class_counts[cls]["hall"] += 1
        else:
            per_class_counts[cls]["wrong"] += 1

        if args.verbose or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(df)}] {img_name}: truth={truth}, pred={pred}, "
                  f"score={sc:+.2f}, time={dt:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    if not results:
        print("[Eval] No results to summarise.")
        return

    total     = len(results)
    correct   = sum(1 for r in results if r["pred"] == r["truth"])
    wrong     = sum(1 for r in results if r["pred"] not in {"5"} and r["pred"] in COMPETITION_OPTS and r["pred"] != r["truth"])
    skipped   = sum(1 for r in results if r["pred"] == "5")
    hallucin  = sum(1 for r in results if r["pred"] not in VALID_ANSWERS)
    total_sc  = sum(r["score"] for r in results)
    accuracy  = correct / total
    max_score = total * 1.0  # If all correct
    avg_time  = total_time / total

    print("\n" + "=" * 60)
    print(" OFFLINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total questions:   {total}")
    print(f"  Correct:           {correct} ({correct/total*100:.1f}%)")
    print(f"  Wrong:             {wrong} ({wrong/total*100:.1f}%)")
    print(f"  Skipped (5):       {skipped} ({skipped/total*100:.1f}%)")
    print(f"  Hallucinated:      {hallucin} ({hallucin/total*100:.1f}%)")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Competition Score: {total_sc:.2f} / {max_score:.2f}")
    print(f"  Score Percentage:  {total_sc/max_score*100:.1f}%")
    print(f"  Avg time/question: {avg_time:.2f}s")
    print(f"  Total time:        {total_time/60:.1f} min")
    print()
    print("  Per-option breakdown:")
    for cls in sorted(per_class_counts.keys()):
        c = per_class_counts[cls]
        tot_cls = sum(c.values())
        print(f"    Option {cls}: correct={c['correct']}, wrong={c['wrong']}, skip={c['skip']}, hall={c['hall']} (total={tot_cls})")
    print("=" * 60)

    # Save results CSV
    if args.save:
        out_path = "offline_eval_results.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"\n[Eval] Detailed results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluator on synthetic dataset")
    parser.add_argument("--config",     default="config.yaml",    help="Path to config.yaml")
    parser.add_argument("--csv",        default="data/synthetic_data/train.csv")
    parser.add_argument("--images_dir", default="data/synthetic_data/images/")
    parser.add_argument("--n",          type=int, default=50,     help="Number of questions to eval (0=all)")
    parser.add_argument("--verbose",    action="store_true",      help="Print every question")
    parser.add_argument("--save",       action="store_true",      help="Save results CSV")
    args = parser.parse_args()
    evaluate(args)
