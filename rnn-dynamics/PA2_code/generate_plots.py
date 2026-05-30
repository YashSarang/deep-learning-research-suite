#!/usr/bin/env python3
"""
CS728 PA2 — Generate all report plots from saved .npz files.

Usage:
    python generate_plots.py                      # generates all available plots
    python generate_plots.py --experiments A1 A4   # only specific experiments
    python generate_plots.py --outdir ./figures    # custom output directory

Produces (per experiment):
  1. Histogram of log10 ||dL/dh_t||  (gradient-through-time)
  2. Histogram of hidden saturation distance
  3. Validation error curve
  4. rho(W_hh) over training
  5. [GRU only] Gate z saturation-distance histogram
  6. [GRU only] Gate r saturation-distance histogram

Produces (comparison figures):
  7. RNN vs GRU validation error overlay (A1 vs A4, A2 vs A5)
  8. No-clip vs clip gradient histograms (A1 vs A2 vs A3)
  9. Combined rho(W_hh) evolution
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ─── Configuration ───────────────────────────────────────────────────────────

# Map experiment name → npz filename (without path)
EXPERIMENTS = {
    "A1": ("A1_mem_rnn_tanh_noclip_final_state.npz",   "A1: RNN tanh, no clip (mem)"),
    "A2": ("A2_mem_rnn_tanh_clip005_final_state.npz",   "A2: RNN tanh, clip=0.05 (mem)"),
    "A3": ("A3_mem_rnn_tanh_clip001_final_state.npz",   "A3: RNN tanh, clip=0.01 (mem)"),
    "A4": ("A4_mem_gru_noclip_final_state.npz",         "A4: GRU, no clip (mem)"),
    "A5": ("A5_mem_gru_clip005_final_state.npz",        "A5: GRU, clip=0.05 (mem)"),
    "B1": ("B1_mul_rnn_tanh_noclip_final_state.npz",    "B1: RNN tanh, no clip (mul)"),
    "B2": ("B2_mul_gru_noclip_final_state.npz",         "B2: GRU, no clip (mul)"),
}

# Style
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "A1": "#e74c3c", "A2": "#3498db", "A3": "#2ecc71",
    "A4": "#9b59b6", "A5": "#f39c12",
    "B1": "#e74c3c", "B2": "#9b59b6",
}

EPS = 1e-12
DIAG_BINS = 60


def load_npz(path):
    """Load npz and return as dict-like object."""
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def get_valid_checkpoints(arr):
    """Return indices of valid (non-sentinel) checkpoint entries."""
    return np.where(arr >= 0)[0]


def get_last_valid_row(arr_2d):
    """Get last row that has at least one finite value."""
    for i in range(arr_2d.shape[0] - 1, -1, -1):
        row = arr_2d[i]
        if np.any(np.isfinite(row) & (row != 0)):
            return row[np.isfinite(row)]
    return np.array([])


# ─── Per-experiment plots ────────────────────────────────────────────────────

def plot_grad_histogram(z, title, outpath):
    """Plot 1: Histogram of log10 ||dL/dh_t|| at the final checkpoint."""
    g = get_last_valid_row(z["grad_time"])
    if len(g) == 0:
        print(f"  [skip] No gradient data for {title}")
        return
    g_log = np.log10(g + EPS)

    fig, ax = plt.subplots()
    ax.hist(g_log, bins=DIAG_BINS, color="#3498db", edgecolor="white", alpha=0.85)
    ax.set_xlabel(r"$\log_{10} \| \partial L / \partial h_t \|_2$")
    ax.set_ylabel("Count (timesteps)")
    ax.set_title(f"Gradient-through-time — {title}")

    # Add statistics annotation
    stats = f"mean={g_log.mean():.2f}\nmed={np.median(g_log):.2f}\np5={np.quantile(g_log,0.05):.2f}\np95={np.quantile(g_log,0.95):.2f}"
    ax.text(0.97, 0.95, stats, transform=ax.transAxes, va="top", ha="right",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7))
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_sat_histogram(z, title, outpath):
    """Plot 2: Histogram of hidden saturation distance at the final checkpoint."""
    s = get_last_valid_row(z["sat_time"])
    if len(s) == 0:
        print(f"  [skip] No saturation data for {title}")
        return

    fig, ax = plt.subplots()
    ax.hist(s, bins=DIAG_BINS, range=(0, 1), color="#e74c3c", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Distance to saturation  $d(h) = 1 - |h|$")
    ax.set_ylabel("Count (timesteps)")
    ax.set_title(f"Hidden saturation distance — {title}")

    sat_frac = (s < 0.05).mean() * 100
    stats = f"mean={s.mean():.3f}\nsat<0.05: {sat_frac:.1f}%"
    ax.text(0.97, 0.95, stats, transform=ax.transAxes, va="top", ha="right",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7))
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_valid_error(z, title, outpath, checkFreq=20):
    """Plot 3: Validation error (%) over training iterations."""
    ve = z["valid_error"]
    valid_idx = get_valid_checkpoints(ve)
    if len(valid_idx) == 0:
        print(f"  [skip] No validation data for {title}")
        return

    iters = valid_idx * checkFreq

    fig, ax = plt.subplots()
    ax.plot(iters, ve[valid_idx], color="#2ecc71", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation Error (%)")
    ax.set_title(f"Validation error — {title}")
    ax.set_ylim(bottom=-2)

    best = ve[valid_idx].min()
    ax.axhline(best, color="gray", linestyle="--", alpha=0.5, label=f"Best: {best:.2f}%")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_rho(z, title, outpath, checkFreq=20):
    """Plot 4: rho(W_hh) over training."""
    rho = z["rho_Whh"]
    valid_idx = get_valid_checkpoints(rho)
    if len(valid_idx) == 0:
        print(f"  [skip] No rho data for {title}")
        return

    iters = valid_idx * checkFreq

    fig, ax = plt.subplots()
    ax.plot(iters, rho[valid_idx], color="#9b59b6", linewidth=1.5)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="ρ = 1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\rho(W_{hh})$")
    ax.set_title(f"Spectral radius — {title}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_gate_sat(z, gate_key, gate_name, title, outpath):
    """Plot 5/6: Gate saturation-distance histogram (GRU only)."""
    if gate_key not in z.files:
        return
    gs = get_last_valid_row(z[gate_key])
    if len(gs) == 0:
        print(f"  [skip] No {gate_name} gate data for {title}")
        return

    fig, ax = plt.subplots()
    color = "#f39c12" if "z" in gate_name.lower() else "#1abc9c"
    ax.hist(gs, bins=DIAG_BINS, range=(0, 0.5), color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel(f"Gate saturation distance  $d({gate_name}) = \\min({gate_name}, 1-{gate_name})$")
    ax.set_ylabel("Count (timesteps)")
    ax.set_title(f"{gate_name}-gate saturation — {title}")

    near0 = (gs < 0.05).mean() * 100
    stats = f"mean={gs.mean():.3f}\nsat<0.05: {near0:.1f}%"
    ax.text(0.97, 0.95, stats, transform=ax.transAxes, va="top", ha="right",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7))
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def generate_per_experiment(name, npz_path, label, outdir, checkFreq=20):
    """Generate all plots for a single experiment."""
    z = load_npz(npz_path)
    if z is None:
        print(f"[SKIP] {name}: file not found ({npz_path})")
        return

    prefix = os.path.join(outdir, name)
    print(f"\n[{name}] {label}")

    plot_grad_histogram(z, label, f"{prefix}_grad_histogram.png")
    plot_sat_histogram(z, label, f"{prefix}_sat_histogram.png")
    plot_valid_error(z, label, f"{prefix}_valid_error.png", checkFreq)
    plot_rho(z, label, f"{prefix}_rho.png", checkFreq)

    # GRU gate plots
    plot_gate_sat(z, "gate_z_sat_time", "z", label, f"{prefix}_gate_z_sat.png")
    plot_gate_sat(z, "gate_r_sat_time", "r", label, f"{prefix}_gate_r_sat.png")


# ─── Comparison plots ────────────────────────────────────────────────────────

def plot_comparison_valid_error(datasets, title, outpath, checkFreq=20):
    """Overlay validation error curves for multiple experiments."""
    fig, ax = plt.subplots()
    for name, z in datasets:
        ve = z["valid_error"]
        vi = get_valid_checkpoints(ve)
        if len(vi) == 0:
            continue
        iters = vi * checkFreq
        color = COLORS.get(name, None)
        label_text = EXPERIMENTS[name][1]
        ax.plot(iters, ve[vi], linewidth=1.5, color=color, label=label_text)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation Error (%)")
    ax.set_title(title)
    ax.set_ylim(bottom=-2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_comparison_grad_histograms(datasets, title, outpath):
    """Overlay gradient-through-time histograms for multiple experiments."""
    fig, ax = plt.subplots()
    for name, z in datasets:
        g = get_last_valid_row(z["grad_time"])
        if len(g) == 0:
            continue
        g_log = np.log10(g + EPS)
        color = COLORS.get(name, None)
        label_text = EXPERIMENTS[name][1]
        ax.hist(g_log, bins=DIAG_BINS, color=color, alpha=0.4, label=label_text, edgecolor="none")

    ax.set_xlabel(r"$\log_{10} \| \partial L / \partial h_t \|_2$")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_comparison_rho(datasets, title, outpath, checkFreq=20):
    """Overlay rho(W_hh) curves for multiple experiments."""
    fig, ax = plt.subplots()
    for name, z in datasets:
        rho = z["rho_Whh"]
        vi = get_valid_checkpoints(rho)
        if len(vi) == 0:
            continue
        iters = vi * checkFreq
        color = COLORS.get(name, None)
        label_text = EXPERIMENTS[name][1]
        ax.plot(iters, rho[vi], linewidth=1.5, color=color, label=label_text)

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="ρ = 1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\rho(W_{hh})$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_comparison_sat_histograms(datasets, title, outpath):
    """Overlay hidden saturation-distance histograms."""
    fig, ax = plt.subplots()
    for name, z in datasets:
        s = get_last_valid_row(z["sat_time"])
        if len(s) == 0:
            continue
        color = COLORS.get(name, None)
        label_text = EXPERIMENTS[name][1]
        ax.hist(s, bins=DIAG_BINS, range=(0, 1), color=color, alpha=0.4,
                label=label_text, edgecolor="none")

    ax.set_xlabel("Distance to saturation")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def generate_comparisons(data_dir, outdir, checkFreq=20):
    """Generate all comparison figures from available data."""
    loaded = {}
    for name, (fname, _) in EXPERIMENTS.items():
        path = os.path.join(data_dir, fname)
        z = load_npz(path)
        if z is not None:
            loaded[name] = z

    print("\n" + "=" * 60)
    print("COMPARISON PLOTS")
    print("=" * 60)

    # ── Task 1 (Memorization): RNN clipping comparison (A1 vs A2 vs A3) ──
    rnn_clip = [(n, loaded[n]) for n in ["A1", "A2", "A3"] if n in loaded]
    if len(rnn_clip) >= 2:
        print("\n[CMP] RNN no-clip vs clip (Memorization)")
        plot_comparison_valid_error(rnn_clip,
            "RNN clipping comparison — Validation Error (Mem)",
            os.path.join(outdir, "CMP_rnn_clip_valid_error.png"), checkFreq)
        plot_comparison_grad_histograms(rnn_clip,
            "RNN clipping comparison — Gradient histograms (Mem)",
            os.path.join(outdir, "CMP_rnn_clip_grad_hist.png"))
        plot_comparison_sat_histograms(rnn_clip,
            "RNN clipping comparison — Saturation (Mem)",
            os.path.join(outdir, "CMP_rnn_clip_sat_hist.png"))
        plot_comparison_rho(rnn_clip,
            "RNN clipping comparison — ρ(W_hh) (Mem)",
            os.path.join(outdir, "CMP_rnn_clip_rho.png"), checkFreq)

    # ── Task 1: GRU clipping comparison (A4 vs A5) ──
    gru_clip = [(n, loaded[n]) for n in ["A4", "A5"] if n in loaded]
    if len(gru_clip) >= 2:
        print("\n[CMP] GRU no-clip vs clip (Memorization)")
        plot_comparison_valid_error(gru_clip,
            "GRU clipping comparison — Validation Error (Mem)",
            os.path.join(outdir, "CMP_gru_clip_valid_error.png"), checkFreq)
        plot_comparison_grad_histograms(gru_clip,
            "GRU clipping comparison — Gradient histograms (Mem)",
            os.path.join(outdir, "CMP_gru_clip_grad_hist.png"))

    # ── Task 1: RNN vs GRU (A1 vs A4, no clipping) ──
    rnn_vs_gru_noclip = [(n, loaded[n]) for n in ["A1", "A4"] if n in loaded]
    if len(rnn_vs_gru_noclip) >= 2:
        print("\n[CMP] RNN vs GRU no-clip (Memorization)")
        plot_comparison_valid_error(rnn_vs_gru_noclip,
            "RNN vs GRU (no clip) — Validation Error (Mem)",
            os.path.join(outdir, "CMP_rnn_vs_gru_noclip_valid.png"), checkFreq)
        plot_comparison_grad_histograms(rnn_vs_gru_noclip,
            "RNN vs GRU (no clip) — Gradient histograms (Mem)",
            os.path.join(outdir, "CMP_rnn_vs_gru_noclip_grad.png"))
        plot_comparison_rho(rnn_vs_gru_noclip,
            "RNN vs GRU (no clip) — ρ(W_hh) (Mem)",
            os.path.join(outdir, "CMP_rnn_vs_gru_noclip_rho.png"), checkFreq)

    # ── Task 1: RNN vs GRU (A2 vs A5, with clipping) ──
    rnn_vs_gru_clip = [(n, loaded[n]) for n in ["A2", "A5"] if n in loaded]
    if len(rnn_vs_gru_clip) >= 2:
        print("\n[CMP] RNN vs GRU with clip (Memorization)")
        plot_comparison_valid_error(rnn_vs_gru_clip,
            "RNN vs GRU (clip=0.05) — Validation Error (Mem)",
            os.path.join(outdir, "CMP_rnn_vs_gru_clip_valid.png"), checkFreq)

    # ── All Memorization experiments together ──
    all_mem = [(n, loaded[n]) for n in ["A1", "A2", "A3", "A4", "A5"] if n in loaded]
    if len(all_mem) >= 2:
        print("\n[CMP] All memorization experiments")
        plot_comparison_valid_error(all_mem,
            "All memorization — Validation Error",
            os.path.join(outdir, "CMP_all_mem_valid_error.png"), checkFreq)

    # ── Task 2 (Multiplication): RNN vs GRU (B1 vs B2) ──
    mul_cmp = [(n, loaded[n]) for n in ["B1", "B2"] if n in loaded]
    if len(mul_cmp) >= 2:
        print("\n[CMP] RNN vs GRU (Multiplication)")
        plot_comparison_valid_error(mul_cmp,
            "RNN vs GRU — Validation Error (Mul)",
            os.path.join(outdir, "CMP_mul_rnn_vs_gru_valid.png"), checkFreq)
        plot_comparison_grad_histograms(mul_cmp,
            "RNN vs GRU — Gradient histograms (Mul)",
            os.path.join(outdir, "CMP_mul_rnn_vs_gru_grad.png"))
        plot_comparison_rho(mul_cmp,
            "RNN vs GRU — ρ(W_hh) (Mul)",
            os.path.join(outdir, "CMP_mul_rnn_vs_gru_rho.png"), checkFreq)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate PA2 report plots")
    parser.add_argument("--datadir", type=str, default=".",
                        help="Directory containing npz files (default: current dir)")
    parser.add_argument("--outdir", type=str, default="./figures",
                        help="Directory to save figures (default: ./figures)")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Experiment names to plot (e.g., A1 A4). Default: all available.")
    parser.add_argument("--checkFreq", type=int, default=20,
                        help="Checkpoint frequency used during training (default: 20)")
    parser.add_argument("--no-comparisons", action="store_true",
                        help="Skip generating comparison plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Determine which experiments to process
    exp_names = args.experiments if args.experiments else list(EXPERIMENTS.keys())

    print("=" * 60)
    print("CS728 PA2 — Report Plot Generator")
    print(f"  Data dir:  {os.path.abspath(args.datadir)}")
    print(f"  Output dir: {os.path.abspath(args.outdir)}")
    print(f"  Experiments: {', '.join(exp_names)}")
    print("=" * 60)

    # Per-experiment plots
    for name in exp_names:
        if name not in EXPERIMENTS:
            print(f"[WARN] Unknown experiment '{name}', skipping.")
            continue
        fname, label = EXPERIMENTS[name]
        npz_path = os.path.join(args.datadir, fname)
        generate_per_experiment(name, npz_path, label, args.outdir, args.checkFreq)

    # Comparison plots
    if not args.no_comparisons:
        generate_comparisons(args.datadir, args.outdir, args.checkFreq)

    print("\n" + "=" * 60)
    n_figs = len([f for f in os.listdir(args.outdir) if f.endswith(".png")])
    print(f"Done! {n_figs} figures saved to {os.path.abspath(args.outdir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
