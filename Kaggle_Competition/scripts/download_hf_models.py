"""
Download HuggingFace Models for Offline Kaggle Usage.

Downloads:
  - Qwen2.5-VL-72B-Instruct (or smaller variant via --vlm_repo)
  - all-MiniLM-L6-v2 sentence embedder

Usage:
    # Full 72B (default):
    python scripts/download_hf_models.py

    # Smaller 7B for faster testing:
    python scripts/download_hf_models.py --vlm_repo Qwen/Qwen2.5-VL-7B-Instruct

    # Custom output dir:
    python scripts/download_hf_models.py --out_dir /path/to/models
"""
import os
import sys
import argparse
from huggingface_hub import snapshot_download

VLM_REPO    = "Qwen/Qwen2.5-VL-72B-Instruct"
EMBED_REPO  = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR     = "models"

# Patterns to ignore (reduce download size)
IGNORE_PATTERNS = [
    "*.msgpack",
    "*.h5",
    "*.ot",            # OpenType fonts
    "flax_model*",     # Flax weights (we use PyTorch)
    "tf_model*",       # TF weights
    "rust_model*",
    "*.gguf",          # GGUF (quantised, we'll do our own BnB quant)
]


def download_model(repo_id: str, local_dir: str, ignore: list[str] | None = None) -> bool:
    """Download a HuggingFace repo to local_dir. Returns True on success."""
    print(f"\n{'─'*60}")
    print(f" Downloading: {repo_id}")
    print(f" Target:      {local_dir}")
    print(f"{'─'*60}")

    os.makedirs(local_dir, exist_ok=True)

    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,   # Kaggle datasets require real files
            ignore_patterns=ignore or [],
            resume_download=True,           # Resume if interrupted
        )
        files = os.listdir(local_dir)
        if not files:
            print(f"[ERROR] {repo_id}: directory is empty after download.")
            return False

        size_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, fns in os.walk(local_dir) for f in fns
        ) / 1e9

        print(f"[OK] Downloaded {len(files)} top-level files ({size_gb:.1f} GB)")
        return True

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Download cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {repo_id}: {e}")
        return False


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nTarget output directory: {args.out_dir}")
    print(f"VLM repo:               {args.vlm_repo}")
    print(f"Embedder repo:          {args.embed_repo}")

    # ── Download VLM ──────────────────────────────────────────────────────
    vlm_dir    = os.path.join(args.out_dir, "vlm")
    vlm_ok     = download_model(args.vlm_repo, vlm_dir, ignore=IGNORE_PATTERNS)

    # ── Download Embedder ─────────────────────────────────────────────────
    embed_dir  = os.path.join(args.out_dir, "embeddings")
    embed_ok   = download_model(args.embed_repo, embed_dir, ignore=["*.ot"])

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(" Download Summary")
    print(f"{'='*60}")
    print(f"  VLM ({args.vlm_repo}):      {'SUCCESS' if vlm_ok else 'FAILED'}")
    print(f"  Embedder ({args.embed_repo}): {'SUCCESS' if embed_ok else 'FAILED'}")

    if not (vlm_ok and embed_ok):
        print("\n[!] One or more downloads failed. Check internet connection and retry.")
        sys.exit(1)

    print("\n[OK] All models ready for offline use.")
    print("\nNext steps:")
    print("  1. python scripts/build_faiss.py")
    print("  2. python scripts/download_wheels.py  (if using custom env)")
    print("  3. Upload kaggle_dataset/ to Kaggle as a private dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models for offline Kaggle usage")
    parser.add_argument("--vlm_repo",   default=VLM_REPO,   help="HuggingFace VLM repo ID")
    parser.add_argument("--embed_repo", default=EMBED_REPO, help="HuggingFace embedder repo ID")
    parser.add_argument("--out_dir",    default=OUT_DIR,    help="Output directory for models")
    args = parser.parse_args()
    main(args)
