# CS728 – Programming Assignment 2

## Running the Code 

Ensure the environment is Python 3.12

```bash

python3.12 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131

pip install -r requirements.txt
```

Copy command from run_all.ps1 to run any desired experiment

or 

Refer to commands.txt for all the commands used to run the experiments

## Training Dynamics of RNNs and GRUs

**Due:** 10 March 2026 (subject to change)  
**Course:** CS728  
**Reference:** Pascanu et al., *On the difficulty of training RNNs*

---

## 1. Assignment Overview

This assignment studies the training dynamics of recurrent neural networks, focusing on:

- Vanishing and exploding gradients
- Activation saturation
- Gate saturation (GRU)
- Spectral radius behavior
- Effect of gradient clipping
- Comparison between Vanilla RNN and GRU

**You will:**
- Implement a Vanilla RNN cell from equations
- Implement a GRU cell from equations
- Train on synthetic long-range dependency tasks
- Log and analyze gradient diagnostics
- Compare behavior across clipping regimes and architectures

---

## 2. Learning Goals

By completing this assignment, you will:

- Understand why RNNs suffer from vanishing/exploding gradients
- Analyze gradient flow through time
- Study hidden state saturation effects
- Compare gating mechanisms in GRUs
- Interpret spectral radius behavior
- Relate diagnostics to learning stability

---

## 3. Repository Structure

```text
PA2_code/
└── trainingRNNs_torch/
    ├── train.py  ← Entry point
    ├── model.py  ← YOU implement RNN and GRU here
    └── tasks.py  ← Task generators (DO NOT MODIFY)
```

**Run training using:**
```bash
python -m trainingRNNs_torch.train [flags]
```

---

## 4. Important Training Notes

- **Iterations** = SGD updates (NOT epochs).
- Each iteration samples a fresh mini-batch.
- There is **NO** dataset epoch.
- No GPU required.
- Use provided command configurations exactly.
- Only modify the `--name` flag for organizing runs.

---

## 5. Implementation Requirements

### Part 1 – Vanilla RNN

Implement the following equations:

$$h_t = \phi(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$
$$o_t = W_{ho}h_t + b_o$$

Where:
- $\phi$ = tanh or sigmoid (based on flag)

**Required:**
- Implement recurrent weight exposure via `rho()` (spectral radius proxy).
- Return:
  - `logits` (task dependent)
  - Full hidden sequence $h_{1:T}$ with shape `(T, B, H)`
- Ensure compatibility with diagnostic hooks.

### Part 2 – GRU

Implement from equations:

$$z_t = \sigma(W_{xz}x_t + W_{hz}h_{t-1} + b_z)$$
$$r_t = \sigma(W_{xr}x_t + W_{hr}h_{t-1} + b_r)$$
$$\tilde{h}_t = \tanh(W_{xh}x_t + W_{hh}(r_t \odot h_{t-1}) + b_h)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Required:**
- Return `logits` + full hidden sequence.
- When `--diagGates` is enabled, return $z_t$ and $r_t$.
- Implement `rho()` returning the candidate recurrent matrix ($W_{hh}$).

---

## 6. Diagnostics You Must Log

At each `--checkFreq`, the script records:

1.  **Global Gradient Norm:** $||\nabla_{\theta} L||_2$
2.  **Post-clipping Gradient Norm**
3.  **Gradient Through Time:** $g_t = ||\partial L / \partial h_t||_2$
    - Plot a histogram of $\log_{10}(g_t)$.
    - Detect vanishing if mass is near -8 to -12.
    - Detect exploding if there is large positive mass.
4.  **Hidden Saturation Distance:**
    - For tanh: $d(h) = 1 - |h|$
    - Small value $\implies$ saturation near $\pm 1$.
5.  **GRU Gate Saturation (if enabled):**
    - $d(v) = \min(v, 1-v)$
    - Near 0 $\implies$ gate saturated; Near 0.5 $\implies$ unsaturated.
    - Plot histograms for $z$-gate and $r$-gate.
6.  **Spectral Radius Proxy:** $\rho(W_{hh})$
    - Track over training.

---

## 7. Required Experimental Runs

You **MUST** run the following experiments:

**Task 1 – Memorization (Classification)**
- **A1:** RNN (no clipping)
- **A2:** RNN (clip 0.05)
- **A3:** RNN (clip 0.01)
- **A4:** GRU (no clipping)
- **A5:** GRU (clip 0.05)

**Task 2 – Multiplication (Regression)**
- **B1:** RNN (no clipping)
- **B2:** GRU (no clipping)

**DO NOT MODIFY:** Hyperparameters, Iterations, Seeds, or Diagnostic settings. Only change `--name`.

---

## 8. Plotting Instructions

Each run saves a `{name}_final_state.npz` file. You should plot:
- Histogram of $\log_{10}(g_t)$
- Saturation histogram
- Validation error curve
- Gate histograms (GRU only)

Use the provided snippet in the assignment for consistent plotting.

---

## 9. Extra Credit

Reproduce successful training on the "temporal order" task using gradient clipping, $\Omega$ regularization, and smart tanh initialization. Find a configuration where training does **NOT** stall at NLL 1.386.

---

## 10. Submission Requirements

Submit the following:
1.  **Code:** `model.py`, `train.py`, and any modified helper files.
2.  **Report (PDF):** For each run (A1–A5, B1–B2), include plots for gradient-through-time, saturation, spectral radius, and provide a technical comparison/analysis.
3.  **Logs:** Raw training logs.
4.  **NPZ Files:** All `{name}_final_state.npz` files.
5.  **Command Log:** A text file listing the exact commands used.
6.  **AI Tool Disclosure (Mandatory):** Link to all AI chats. AI tools are for conceptual help only; code generation is prohibited.

---

## 11. Grading Criteria

1.  **Correct Implementation (Major Weight):** RNN/GRU equations, tensor shapes, diagnostics.
2.  **Completeness of Runs:** All required experiments executed with correct flags.
3.  **Diagnostic Interpretation:** Vanishing/exploding detection, saturation analysis.
4.  **Report Quality:** Clear plots, technical reasoning, insightful comparisons.
5.  **Reproducibility:** Logs match claims, commands documented.

---

## 12. Restrictions

- **MAY NOT:** Use `torch.nn.RNN`, `torch.nn.GRU`, modify task generation, or change hyperparameters.
- **MAY:** Refactor your own code, add helper functions, use AI for conceptual clarification (disclose it).

---

## 13. Recommended Workflow

1.  Implement RNN and validate shapes/forward pass.
2.  Implement GRU and enable diagnostics.
3.  Run A1 (baseline) and then clipping regimes.
4.  Run GRU experiments.
5.  Analyze `.npz` files and generate plots.
6.  Write the report.

---

## 14. Common Mistakes to Avoid

- Returning only the final hidden state (must return full sequence).
- Incorrect tensor shape: use `(T, B, H)` instead of `(B, T, H)`.
- Forgetting to implement `rho()`.
- Not enabling `--diagGates` or `--collectDiags`.
- Ignoring spectral radius logging.

---
