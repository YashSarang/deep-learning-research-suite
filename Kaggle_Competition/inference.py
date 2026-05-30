"""
GNR638 — Kaggle Competition Inference Script
============================================
Entry point for the grading system.

Usage:
    python inference.py --test_dir <absolute_path_to_test_directory>

The test directory must follow the structure:
    <test_dir>/
    ├── test.csv          # columns: image_name  (no .png extension)
    └── images/
        ├── image_1.png
        └── image_2.png
        ...

Outputs:
    submission.csv        # written to current working directory
    columns: image_name, option   (option ∈ {1,2,3,4,5})
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GNR638 VLM-based MCQ solver — batch inference"
    )
    parser.add_argument(
        "--test_dir",
        required=True,
        help="Absolute path to the test directory containing test.csv and images/",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in current directory)",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Output CSV filename (default: submission.csv)",
    )
    return parser.parse_args()


# ── Path helpers ──────────────────────────────────────────────────────────────

def locate_file(candidates: list[str]) -> str:
    """Return the first existing path from a list of candidates."""
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not locate file. Tried:\n" + "\n".join(f"  {c}" for c in candidates)
    )


def find_config(cli_config: str) -> str:
    """Find config.yaml relative to the script or CWD."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return locate_file([
        cli_config,
        os.path.join(script_dir, "config.yaml"),
        os.path.join(os.getcwd(), "config.yaml"),
    ])


# ── Main inference loop ───────────────────────────────────────────────────────

def run_inference(test_dir: str, config_path: str, output_path: str) -> None:
    """
    Full batch inference loop.

    Args:
        test_dir:    Directory with test.csv and images/
        config_path: Path to config.yaml
        output_path: Where to write submission.csv
    """
    # Validate test directory
    if not os.path.isdir(test_dir):
        print(f"[ERROR] test_dir not found: {test_dir}")
        sys.exit(1)

    csv_path = os.path.join(test_dir, "test.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] test.csv not found in: {test_dir}")
        sys.exit(1)

    images_dir = os.path.join(test_dir, "images")
    if not os.path.isdir(images_dir):
        print(f"[ERROR] images/ directory not found in: {test_dir}")
        sys.exit(1)

    # Add project root to sys.path so src/ is importable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    print("=" * 60)
    print(" GNR638 — MCQ Solver Inference")
    print("=" * 60)
    print(f"  Test directory : {test_dir}")
    print(f"  Config         : {config_path}")
    print(f"  Output         : {output_path}")
    print("=" * 60)

    # Load test CSV
    df = pd.read_csv(csv_path)
    if "image_name" not in df.columns:
        print("[ERROR] test.csv must contain an 'image_name' column.")
        sys.exit(1)

    total = len(df)
    print(f"\n[Inference] Found {total} questions in test.csv")

    # Initialise models (loads 72B VLM into GPU — takes ~2 min)
    print("\n[Inference] Initialising models (may take 2–5 minutes for 72B)...")
    try:
        from src.pipeline import initialize_models, solve
        initialize_models(config_path)
    except Exception as e:
        print(f"\n[FATAL] Model initialisation failed: {e}")
        print("        Ensure models are downloaded: python scripts/download_hf_models.py")
        sys.exit(1)

    # Run inference
    results = []
    total_time = 0.0

    print(f"\n[Inference] Running on {total} images...\n")
    for idx, row in df.iterrows():
        img_name = str(row["image_name"]).strip()
        img_path = os.path.join(images_dir, f"{img_name}.png")

        if not os.path.exists(img_path):
            print(f"  [{idx + 1}/{total}] WARNING — image not found: {img_path} → defaulting to 5 (skip)")
            results.append({"image_name": img_name, "option": 5})
            continue

        t0 = time.time()
        try:
            answer = solve(img_path, config_path=config_path)
        except Exception as e:
            print(f"  [{idx + 1}/{total}] ERROR solving {img_name}: {e} → defaulting to 5 (skip)")
            answer = "5"
        dt = time.time() - t0
        total_time += dt

        results.append({"image_name": img_name, "option": int(answer)})

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            avg_t = total_time / (idx + 1)
            remaining = (total - idx - 1) * avg_t / 60
            print(
                f"  [{idx + 1}/{total}] {img_name}: option={answer}  "
                f"({dt:.1f}s | avg {avg_t:.1f}s | ~{remaining:.0f} min left)"
            )

    # Write submission.csv to current working directory (not test_dir)
    out_df = pd.DataFrame(results)[["image_name", "option"]]
    out_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f" Inference complete.")
    print(f"  Questions solved : {len(results)}")
    print(f"  Total time       : {total_time / 60:.1f} min")
    print(f"  Output written   : {os.path.abspath(output_path)}")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Resolve config path
    try:
        config_path = find_config(args.config)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    run_inference(
        test_dir=args.test_dir,
        config_path=config_path,
        output_path=args.output,
    )
