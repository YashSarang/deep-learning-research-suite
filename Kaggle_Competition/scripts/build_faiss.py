"""
Build FAISS Index from Knowledge Base Text Files.

Run once on the server before running inference:
    python scripts/build_faiss.py

Output:
    models/faiss/index.faiss
    models/faiss/metadata.pkl
"""
import os
import pickle
import argparse
import numpy as np

CHUNK_SIZE    = 3   # sentences per chunk
EMBED_MODEL   = "all-MiniLM-L6-v2"
KB_DIR        = "src/retrieval/knowledge_base"
OUTPUT_DIR    = "models/faiss"


def load_text_files(kb_dir: str) -> list[str]:
    """Load and split all .txt files in KB dir into chunks."""
    import re
    chunks: list[str] = []
    for fname in sorted(os.listdir(kb_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(kb_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        # Split on newlines, filter empty
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Group lines into chunks
        for i in range(0, len(lines), CHUNK_SIZE):
            chunk = " ".join(lines[i : i + CHUNK_SIZE])
            if len(chunk) > 20:  # Skip trivially short chunks
                chunks.append(chunk)
        print(f"  Loaded {fname}: {len(lines)} lines")
    return chunks


def build_faiss_index(chunks: list[str], embed_model_path: str) -> tuple:
    """Embed chunks and build FAISS flat index."""
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"[FAISS] Loading embedder from: {embed_model_path}")
    model = SentenceTransformer(embed_model_path, device="cpu")

    print(f"[FAISS] Embedding {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
        show_progress_bar=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    print(f"[FAISS] Embedding dim: {dim}")

    # IP (inner product) index = cosine similarity on normalized vectors
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[FAISS] Index built with {index.ntotal} vectors.")

    return index, chunks  # metadata is just the original text chunks


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("[FAISS] Loading knowledge base text files...")
    chunks = load_text_files(args.kb_dir)
    print(f"[FAISS] Total chunks: {len(chunks)}")

    index, metadata = build_faiss_index(chunks, args.embed_model)

    # Save
    import faiss
    index_path = os.path.join(args.output_dir, "index.faiss")
    meta_path  = os.path.join(args.output_dir, "metadata.pkl")

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[FAISS] Saved index to: {index_path}")
    print(f"[FAISS] Saved metadata to: {meta_path}")
    print(f"[FAISS] Done! {index.ntotal} vectors indexed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build offline FAISS index from DL knowledge base")
    parser.add_argument("--kb_dir",      default=KB_DIR,      help="Knowledge base directory")
    parser.add_argument("--embed_model", default="models/embeddings",
                        help="Path to local all-MiniLM-L6-v2 weights")
    parser.add_argument("--output_dir",  default=OUTPUT_DIR,  help="Output directory for FAISS files")
    args = parser.parse_args()
    main(args)
