# Assignment 3: Retrieval, Attention, and LLMs

This repository contains the optimized implementation for the three-part Programming Assignment 3 on Retrieval, Attention, and LLMs.

## Prerequisites & Hugging Face Authentication
The code uses `meta-llama/Llama-3.2-1B-Instruct` which is a **gated model**. Before running any script, you must:
1. Accept the LLaMA 3.2 license on Hugging Face (https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
2. Authenticate locally by running: `huggingface-cli login` and providing your gated-access token.

## Provided Scripts

### 1. `run1.py` (Classical Retrieval)
Runs sparse (BM25) and dense (msmarco-MiniLM-L-6-v3 & UAE-Large-V1) baselines over the 5000 test queries.
- **Runtime:** ~1-2 minutes.
- **Usage:** `python run1.py`

### 2. `run_all.py` (Highly Optimized Part 2 & Part 3)
***Why a combined script?***
A standard `output_attentions=True` pass over 3000 sequence tokens generates a $16 \times 32 \times 3000 \times 3000$ matrix that consumes roughly **9.5 GB of VRAM**. On consumer hardware (like an 8GB RTX 4060 GPU), this causes the system to natively spill into unified CPU memory, choking inference times to ~3 seconds per query, which equals a painful **~8 hours** total execution time if Parts 2 & 3 run sequentially.

`run_all.py` utilizes **PyTorch Forward Hooks** on `LlamaAttention` modules to incrementally compute the average query attention (for Part 2) and extract the designated top 20 specific heads (for Part 3) *during the forward pass*, immediately dropping the bulk array from VRAM. This restricts total VRAM to <6 GB, preventing memory swapping and combining the entire Part 2 and Part 3 evaluation pass into a **single 5000-query loop**.

- **Phase 1 (Code3 / Head Selection):** Uses training queries to identify the predictive retrieval heads using both MRR and Attention-Mass criteria (Bonus).
- **Phase 2 (Evaluation):** Passes the 5000 test queries through the model **once**, executing Part 2 metrics (global query correlation) and all Bonus Part 3 configurations (`max_heads=10, 20, 30` MRR, and `max_heads=20` Attn-Mass) concurrently.
- **Runtime:** ~3.5 hours for the unified 5000-sample pass.
- **Outputs generated:**
    - `plot2/gold_attention_plot.png` (Lost-in-the-middle plot)
    - `results/part2_results.json` (Recall@1/5 for full attention)
    - `results/part3_bonus_results.json` (The chosen heads and Recall metrics for all requested Bonus configurations)

- **Usage:** `python run_all.py`

### 3. `generate_report.py` (Automated Markdown Builder)
To fulfill the explicit DevNotes format grading requirements, this newly provided script compiles all the output `.json` results and structures them into a perfectly valid, copy-pasteable markdown file called `final_report.md`. It generates your deliverables instantly.

### 4. `render_report.js` (Premium PDF Generator)
A specialized rendering pipeline using **Puppeteer** and **Marked** to convert `final_report.md` into a high-quality academic PDF with custom CSS (`report_style.css`).
- **Usage:** `node render_report.js`

## Replication & Results Verification

To replicate the results exactly as reported in the final PDF:

1. **Setup Environment:**
   ```bash
   cd CS728_PA3
   pip install -r requirements.txt
   npm install puppeteer marked  # For PDF rendering
   ```
2. **Authenticate with Hugging Face:**
   ```bash
   huggingface-cli login  # Requires access to LLaMA 3.2 gated model
   ```
3. **Execute Master Pipeline:**
   ```powershell
   ./run_pipeline.ps1
   ```
   *This script runs Part 1, Parts 2 & 3 (optimized hooks), and generates the `final_report.md`.*
4. **Generate Final PDF:**
   ```bash
   node render_report.js
   ```

## Deliverables & Report Checklist
When assembly for submission is required:
1. Run the master wrapper `run_pipeline.ps1` to automatically execute all scripts sequentially.
2. Read the resulting `CS728_PA3/final_report.md` document, which has extracted all generated figures (`plot2/*`) and formatted the correct tables referencing Parts 1, 2, and 3.
3. Generate the PDF: `CS728_PA3_Final_Report.pdf`.
4. Compress output files into `[RollNumber].zip` as required.

**Author:** Yash Sarang
