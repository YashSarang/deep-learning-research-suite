# make_report.py
import os
import glob
import csv
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

ASSETS = Path("assets")
OUTPUTS = Path("outputs")
DATA = Path("data")

# ---- Edit these to match your chosen words ----
# must be the SAME 3 used in Task 1 & 2
TASK1_TASK2_WORDS = ["united", "city", "president"]
TASK5_WORDS = ["city", "run", "music", "london",
               "market"]  # 5 diverse words (if Task 5 done)

# ---- Paste your shareable chat links here (required by policy) ----
GENAI_CHAT_LINKS = [
    # "https://chatgpt.com/share/....",
]


def read_csv(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md_table(rows, cols):
    if not rows:
        return "_(not available)_"
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def find_loss_plots():
    if not ASSETS.exists():
        return []
    # Prefer d200 plots if you named them that way
    plots = sorted(glob.glob(str(ASSETS / "glove_loss_*d200*.png")))
    if not plots:
        plots = sorted(glob.glob(str(ASSETS / "glove_loss_*.png")))
    return plots


def file_exists(p: Path) -> bool:
    return p is not None and Path(p).exists()


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    glove_results = read_csv(ASSETS / "glove_search_results.csv")
    mlp_results = read_csv(ASSETS / "task4_mlp_results.csv")
    task5_neighbors = (ASSETS / "task5_neighbors_raw_vs_tfidf.csv")
    task5_metrics = (ASSETS / "task5_tfidf_mlp_metrics.csv")

    loss_plots = find_loss_plots()

    # Try infer best MLP row (highest test_macro_f1)
    best_mlp = None
    if mlp_results:
        def f(x):
            try:
                return float(x.get("test_macro_f1", -1))
            except:
                return -1
        best_mlp = max(mlp_results, key=f)

    report_md = []
    report_md.append(f"# Programming Assignment 1 Report\n")
    report_md.append(f"_Generated draft: {now}_\n")
    report_md.append("---\n")

    # 1 Executive Summary
    report_md.append("## 1. Executive Summary\n")
    report_md.append(
        "This report compares unsupervised word embedding methods (GloVe vs. SVD) trained on a provided CC-News subset "
        "and evaluates their effectiveness on Named Entity Recognition (NER) using CoNLL-2003. We also compare against "
        "a CRF baseline using hand-crafted lexical/shape/subword features.\n"
    )
    if best_mlp:
        report_md.append(
            f"Best MLP configuration (by test Macro-F1): **{best_mlp.get('algo', '?').upper()}**, "
            f"**d={best_mlp.get('d', '?')}**, test Macro-F1={best_mlp.get('test_macro_f1', '?')}, "
            f"test Acc={best_mlp.get('test_acc', '?')}.\n"
        )
    report_md.append("\n---\n")

    # 2 Data & Protocol
    report_md.append("## 2. Data & Evaluation Protocol\n")
    report_md.append(
        "- **Unsupervised training data:** Provided CC-News subset (~67k documents) and provided vocabulary (~25k tokens).\n")
    report_md.append(
        "- **Constraint:** GloVe and SVD training performed only on provided vocabulary (OOV ignored during training).\n")
    report_md.append(
        "- **NER data:** CoNLL-2003 (train/validation/test splits as provided).\n")
    report_md.append(
        "- **Metrics:** Token-level Accuracy and Macro-F1; final metrics reported on **test** split.\n")
    report_md.append("\n---\n")

    # 3 Task 1
    report_md.append("## 3. Task 1 — GloVe Pre-training\n")
    report_md.append("### 3.1 Objective and Setup\n")
    report_md.append(
        "GloVe is trained using a weighted least-squares objective on a global word–word co-occurrence matrix built with a context window `w`.\n")
    report_md.append("### 3.2 Hyperparameter Search (fixed d=200)\n")
    if glove_results:
        report_md.append("Search results:\n\n")
        report_md.append(md_table(
            glove_results,
            cols=["w", "lr", "iters", "x_max", "alpha",
                  "final_loss", "total_train_s", "nnz"]
        ))
        report_md.append("\n")
    else:
        report_md.append(
            "_glove_search_results.csv not found. Add it to assets/._\n")

    report_md.append("\n### 3.3 Loss Curves\n")
    if loss_plots:
        for p in loss_plots:
            report_md.append(f"![GloVe Loss Curve]({p})\n")
    else:
        report_md.append(
            "_No loss plots found in assets/. Expected assets/glove_loss_*.png_\n")

    report_md.append("\n### 3.4 Nearest Neighbors (Top-5)\n")
    report_md.append(
        f"Words used (must match Task 2): **{', '.join(TASK1_TASK2_WORDS)}**\n\n")
    report_md.append(
        "_Paste your printed neighbor lists from task_1.py here (Top-5 for each word)._ \n")
    report_md.append("\n---\n")

    # 4 Task 2
    report_md.append("## 4. Task 2 — SVD Pipeline\n")
    report_md.append(
        "SVD is applied to a sparse term-document matrix X. Token representations are computed as $U_k\\Sigma_k$.\n")
    report_md.append("\n### 4.1 Nearest Neighbors (Top-5)\n")
    report_md.append(
        f"Same 3 words as Task 1: **{', '.join(TASK1_TASK2_WORDS)}**\n\n")
    report_md.append(
        "_Paste your printed neighbor lists from task_2.py here (Top-5 for each word)._ \n")
    report_md.append("\n---\n")

    # 5 Task 3
    report_md.append("## 5. Task 3 — CRF Baseline (Feature Engineering)\n")
    report_md.append("### 5.1 Feature Set\n")
    report_md.append(
        "_List every feature included in the CRF here (copy from your implementation)._ \n")
    report_md.append("\n### 5.2 Test Results and Feature Importance\n")
    report_md.append(
        "_Paste test Accuracy, Macro-F1, and top weighted CRF features here (from task_3.py output)._ \n")
    report_md.append("\n---\n")

    # 6 Task 4
    report_md.append("## 6. Task 4 — MLP (Feature Learning with Embeddings)\n")
    report_md.append("### 6.1 Architecture\n")
    report_md.append(
        "MLP takes a single token embedding as input and predicts one of 9 NER tags.\n")
    report_md.append("\n### 6.2 OOV Strategy\n")
    report_md.append(
        "_Describe and justify your OOV strategy (e.g., longest prefix/suffix match → <UNK> fallback)._ \n")

    report_md.append("\n### 6.3 Results Table (All Runs)\n")
    if mlp_results:
        report_md.append(md_table(
            mlp_results,
            cols=["algo", "d", "oov_mode", "test_acc",
                  "test_macro_f1", "best_dev_macro_f1"]
        ))
        report_md.append("\n")
    else:
        report_md.append(
            "_task4_mlp_results.csv not found. Add it to assets/._\n")

    report_md.append("\n### 6.4 Best Configuration and Comparison vs CRF\n")
    if best_mlp:
        report_md.append(
            f"Best MLP: **{best_mlp.get('algo', '?').upper()}**, d={best_mlp.get('d', '?')}, "
            f"test Macro-F1={best_mlp.get('test_macro_f1', '?')}, test Acc={best_mlp.get('test_acc', '?')}.\n"
        )
    report_md.append(
        "_Compare the best MLP vs CRF and explain why one outperformed the other._\n")
    report_md.append("\n---\n")

    # 7 Task 5
    report_md.append("## 7. Task 5 — Extra Credit (TF-IDF + SVD)\n")
    if task5_neighbors.exists():
        report_md.append(
            "### 7.1 Quality Check 1: Neighbors (Raw vs TF-IDF)\n")
        report_md.append(f"Words: **{', '.join(TASK5_WORDS)}**\n\n")
        report_md.append(f"_See CSV: `{task5_neighbors}`_\n\n")
    else:
        report_md.append(
            "_Task 5 neighbors CSV not found (skip if not doing extra credit)._ \n")

    if task5_metrics.exists():
        report_md.append(
            "### 7.2 Quality Check 2: MLP using TF-IDF SVD vectors\n")
        report_md.append(f"_See CSV: `{task5_metrics}`_\n\n")
    else:
        report_md.append(
            "_Task 5 metrics CSV not found (skip if not doing extra credit)._ \n")

    report_md.append("\n---\n")

    # 8 GenAI disclosure
    report_md.append("## 8. GenAI Usage Disclosure\n")
    report_md.append(
        "This project used an LLM assistant during implementation. Per course policy, the usage is disclosed here with shareable links.\n\n"
    )
    if GENAI_CHAT_LINKS:
        for link in GENAI_CHAT_LINKS:
            report_md.append(f"- {link}\n")
    else:
        report_md.append(
            "_Add shareable chat links here before submitting._\n")

    report_md.append("\n")

    out_md = Path("report.md")
    out_md.write_text("\n".join(report_md), encoding="utf-8")
    print(f"Wrote {out_md}")

    # Optional: convert to PDF using pandoc if available
    if shutil.which("pandoc"):
        out_pdf = Path("report.pdf")
        cmd = ["pandoc", "report.md", "-o", str(out_pdf)]
        try:
            subprocess.check_call(cmd)
            print(f"Wrote {out_pdf}")
        except Exception as e:
            print("Pandoc PDF conversion failed:", e)
            print("You can still submit report.md or convert manually.")
    else:
        print("pandoc not found; skipping PDF build. Convert report.md to PDF manually if needed.")


if __name__ == "__main__":
    main()
