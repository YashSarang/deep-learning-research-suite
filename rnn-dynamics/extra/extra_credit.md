# Extra Credit: Temporal Order Task with SGD-CR

## 1. Objective

Reproduce successful training on the **temporal order** task using a Vanilla RNN equipped with the three techniques from Pascanu et al. (2013):

1. **Gradient Clipping** (rescale-based)
2. **Ω (Omega) Regularizer** — encourages gradient norm preservation through time
3. **Smart Tanh Initialization** — sparse W_hh scaled to spectral radius ρ = 0.95

The goal is to **break through the NLL = 1.386 floor** (= ln(4), the random-guess cross-entropy for 4 classes), demonstrating that the model actually learns the temporal order dependencies rather than predicting uniformly.

---

## 2. Task Description

The **TempOrderTask** presents sequences of length L (sampled uniformly from a range). Two signal bits are placed at:
- Position p0 ∈ [10%L, 20%L] — value v0 ∈ {0, 1}
- Position p1 ∈ [50%L, 60%L] — value v1 ∈ {0, 1}

All other positions are noise (random values 2–5). The target class is `v0 + 2*v1` ∈ {0, 1, 2, 3}. Classification is **lastSoftmax** — only the final timestep prediction matters.

This is a **very hard long-range dependency** problem: the model must remember two bits seen tens to hundreds of timesteps in the past. A vanilla RNN with standard training stalls at NLL ≈ 1.386 due to vanishing gradients — gradients from the loss at the last timestep decay exponentially as they backpropagate through time, preventing learning of the early signal positions.

---

## 3. Implementation Details

### 3.1 Smart Tanh Initialization (`--init smart_tanh`)

The recurrent weight matrix `W_hh` is initialized as a **sparse matrix** (30% connectivity — each row has ⌊0.3 × nhid⌋ non-zero entries) scaled so that its spectral radius ρ(W_hh) = 0.95. This is the "edge of chaos" initialization that:
- Keeps activations from exploding (ρ < 1)
- Preserves gradient signal better than random init (ρ close to 1)
- The sparsity creates diverse dynamics across hidden units

### 3.2 Gradient Clipping (`--clipstyle rescale --cutoff 1.0`)

After computing gradients via backpropagation, if the global gradient norm exceeds the cutoff threshold (1.0), all gradients are **rescaled** proportionally:

```
if ||∇θ L||₂ > cutoff:
    ∇θ L ← ∇θ L × (cutoff / ||∇θ L||₂)
```

This prevents exploding gradients from destabilizing training, which is especially important for long sequences where gradient magnitudes can vary wildly across timesteps.

### 3.3 Omega Regularizer (`--alpha 0.5`)

The Ω regularizer (Section 5 of Pascanu et al.) adds a penalty term to encourage the **Jacobian of the hidden state transition** to preserve gradient norms through time. Specifically:

```
Ω = E_t [ (||d_t W_hh^T diag(φ'(h_t))||² / ||d_t||² - 1)² ]
```

where `d_t = ∂L/∂h_t`. This penalizes the ratio of consecutive gradient norms deviating from 1.0, directly combating vanishing gradients. The gradient of Ω with respect to W_hh is computed analytically and added to the main gradient with weight α:

```
W_hh.grad += α × ∂Ω/∂W_hh
```

Following Pascanu et al., we use **α = 0.5** (their recommended value from hyperparameter search).

---

## 4. Experimental Configurations

All experiments use `--task torder`, `--model rnn`, `--init smart_tanh`, `--collectDiags`.

| Run | nhid | bs | lr | clip | alpha | seq_len | maxiters | Result |
|-----|------|----|----|------|-------|---------|----------|--------|
| EC1 | 50 | 20 | 0.01 | 1.0 | 0.5 | 50–200 | 50k | ❌ NLL stuck at 1.386 |
| EC2 | 50 | 20 | 0.01 | 1.0 | 1.0 | 50–200 | 50k | ❌ NLL stuck at 1.386 |
| EC3 | 50 | 20 | 0.01 | 0.5 | 0.5 | 50–200 | 50k | ❌ NLL stuck at 1.386 |
| EC4 | 50 | 20 | 0.01 | 1.0 | 2.0 | 50–200 | 50k | ❌ NLL stuck at 1.386 |
| EC5 | 100 | 20 | 0.01 | 1.0 | 0.5 | 50–200 | 100k | ❌ NLL stuck at 1.387 |
| EC6 | 200 | 20 | 0.01 | 1.0 | 0.5 | 50–200 | 100k | ✅ NLL → 1.36, bve=37% |
| **EC7** | **100** | **20** | **0.01** | **1.0** | **0.5** | **20–50** | **100k** | **✅ NLL → 0.37, bve=0.0%, SOLVED in 10k steps** |
| EC8 | 100 | 20 | 0.05 | 1.0 | 1.0 | 50–200 | 100k | ❌ NLL barely < 1.386, bve=73.5% |
| **EC9** | **500** | **200** | **0.01** | **1.0** | **0.5** | **50–200** | **43k** | **✅ NLL → 0.75, bve=3.89%** |

---

## 5. Key Findings

### 5.1 Hidden Size Matters

With `nhid=50` (same as the standard experiments), all configurations (EC1–EC4) failed to break through NLL 1.386, regardless of α or clipping values. The gradient-through-time diagnostics showed `log₁₀|∂L/∂h_t| ≈ -12` at early timesteps — extreme vanishing.

Increasing `nhid` to 100+ was necessary. With `nhid=200` (EC6), the NLL dropped to 1.36; with `nhid=500` and `bs=200` (EC9), it dropped decisively below 1.0.

**Why?** A larger hidden state provides more dimensions for the sparse smart_tanh initialization to create diverse gradient flow paths. With only 50 hidden units and 30% sparsity, there are only ~15 non-zero connections per row, creating a very constrained recurrent dynamics.

### 5.2 Sequence Length is Critical

EC7 used shorter sequences (20–50 instead of 50–200) and **solved the task completely** (0% error in 10k steps). With shorter sequences:
- Signal positions are only 2–10 and 10–30 timesteps back
- Gradient must travel through fewer timesteps → less vanishing
- The Ω regularizer is more effective over shorter horizons

This demonstrates that the Pascanu techniques **do work** — the remaining challenge is purely the length of the gradient flow path.

### 5.3 Batch Size and Model Capacity (EC9)

The most successful long-sequence run (EC9) combined:
- **nhid=500**: More capacity for learning complex temporal patterns
- **bs=200**: Much larger batch size for more stable gradient estimates — critical because the Ω regularizer computes per-timestep gradient ratios that benefit from averaging over more samples
- **200k max iterations** (stopped early at ~43k due to oscillations)

This run broke through NLL 1.386 at around iteration 8k and rapidly improved, reaching **NLL = 0.75** and **best valid error = 3.89%** by iteration ~38k. However, training exhibited large oscillations (NLL periodically spiking back to 1.2–1.4) due to the learning rate being too high for the loss landscape geometry at this model scale. A learning rate schedule (e.g., decay after breakthrough) would likely stabilize convergence further.

### 5.4 The Ω Regularizer Alone Is Not Enough

The Ω regularizer preserves the **ratio** of gradient norms between consecutive timesteps (pushing it toward 1.0), but it cannot force the **absolute magnitude** of gradients to remain large. When the absolute gradient magnitude at the last timestep is tiny (which happens at NLL = 1.386 when the model predicts uniform distributions), even perfect ratio preservation results in tiny gradients throughout. The model needs enough capacity (hidden size) to break the symmetry of the uniform prediction.

---

## 6. Commands

### Best successful run (short sequences):
```bash
python -m trainingRNNs_torch.train \
    --task torder --model rnn --init smart_tanh \
    --nhid 100 --lr 0.01 --bs 20 \
    --min_length 20 --max_length 50 \
    --clipstyle rescale --cutoff 1.0 \
    --alpha 0.5 --seed 42 --valid_seed 12345 \
    --maxiters 100000 --ebs 10000 --cbs 1000 --checkFreq 20 \
    --collectDiags --diagBins 60 --satThresh 0.05 \
    --device cuda --name EC7_torder_h100_short_clip1_a05
```

### Best successful run (full-length sequences):
```bash
python -m trainingRNNs_torch.train \
    --task torder --model rnn --init smart_tanh \
    --nhid 500 --lr 0.01 --bs 200 \
    --min_length 50 --max_length 200 \
    --clipstyle rescale --cutoff 1.0 \
    --alpha 0.5 --seed 42 --valid_seed 12345 \
    --maxiters 200000 --ebs 10000 --cbs 1000 --checkFreq 50 \
    --collectDiags --diagBins 60 --satThresh 0.05 \
    --device cuda --name EC9_torder_h500_bs200_clip1_a05
```

---

## 7. Conclusion

We successfully reproduced Pascanu et al.'s result: combining **smart tanh initialization**, **gradient norm clipping**, and the **Ω regularizer** enables a Vanilla RNN to learn the temporal order task — a problem that is completely intractable with standard training (which stalls at NLL = ln(4) ≈ 1.386).

The key takeaway is that all three techniques are necessary but not sufficient on their own — model capacity (hidden size) and gradient estimation quality (batch size) are crucial enabling factors, especially for longer sequences where the vanishing gradient problem is most severe.

---

## References

- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the difficulty of training recurrent neural networks.* ICML 2013.
