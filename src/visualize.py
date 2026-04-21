"""
src/visualize.py
----------------
All plots required by the case study:

  1. Gate Distribution Histogram — shows the "spike at 0" for a pruned model
  2. Training Curves            — accuracy & loss with 4-phase markers
  3. Sparsity vs. Accuracy Bar  — comparison across all lambda runs
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import torch

from src.model import PrunableCNN


# ─── Shared style ─────────────────────────────────────────────────────────────
PHASE_COLORS = {
    "warmup":   "#4C9BE8",
    "sparsify": "#E8914C",
    "finetune": "#5CBF7A",
}
PHASE_LABELS = {
    "warmup":   "Warm-up",
    "sparsify": "Sparsification",
    "finetune": "Fine-tuning",
}
STYLE = {
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#3A3D4D",
    "axes.labelcolor":  "#DADDE6",
    "xtick.color":      "#8B8FA8",
    "ytick.color":      "#8B8FA8",
    "grid.color":       "#2A2D3D",
    "text.color":       "#DADDE6",
    "font.family":      "DejaVu Sans",
}


def _apply_style():
    plt.rcParams.update(STYLE)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Gate Distribution Histogram
# ══════════════════════════════════════════════════════════════════════════════

def plot_gate_histogram(model: PrunableCNN, out_path: str, lam: float) -> None:
    """
    ✅ Checklist: histogram showing the 'spike at 0'.

    Plots the distribution of all sigmoid(gate_scores) values.
    A well-pruned model will show a large spike near 0 (pruned weights) and
    a smaller cluster near 1 (active weights).
    """
    _apply_style()

    gates = model.get_all_gates().numpy()
    total  = len(gates)
    near_0 = (gates < 0.05).sum()
    near_1 = (gates > 0.95).sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])

    # Main histogram
    n, bins, patches = ax.hist(
        gates, bins=100, range=(0, 1),
        color="#4C9BE8", edgecolor="none", alpha=0.85
    )

    # Highlight the spike-at-0 region
    for patch, left in zip(patches, bins[:-1]):
        if left < 0.05:
            patch.set_facecolor("#E84C6B")
            patch.set_alpha(0.95)

    ax.set_title(
        f"Gate Value Distribution  (λ = {lam})\n"
        f"Spike at 0: {near_0/total*100:.1f}% pruned  |  "
        f"Near 1: {near_1/total*100:.1f}% active",
        fontsize=13, pad=14,
    )
    ax.set_xlabel("Gate Value  sigmoid(gate_scores)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", linewidth=0.5)

    # Legend
    red_patch  = mpatches.Patch(color="#E84C6B", label=f"Pruned (< 0.05): {near_0:,}")
    blue_patch = mpatches.Patch(color="#4C9BE8", label=f"Active (≥ 0.05): {total - near_0:,}")
    ax.legend(handles=[red_patch, blue_patch], fontsize=10, framealpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Gate histogram → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Training Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: list[dict], out_path: str, lam: float) -> None:
    """
    Training accuracy + loss over epochs.
    Phase transitions are marked with vertical dashed lines and shaded regions.
    """
    _apply_style()

    epochs      = [r["epoch"]     for r in history]
    train_acc   = [r["train_acc"] for r in history]
    test_acc    = [r["test_acc"]  for r in history]
    train_loss  = [r["train_loss"] for r in history]
    test_loss   = [r["test_loss"]  for r in history]
    phases      = [r["phase"]      for r in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.patch.set_facecolor(STYLE["figure.facecolor"])
    fig.suptitle(f"Training Curves — λ = {lam}", fontsize=14, y=0.98)

    # ── Shade phases ─────────────────────────────────────────────────────────
    def _shade_phases(ax):
        prev_phase = None
        start = 0
        for i, (ep, ph) in enumerate(zip(epochs, phases)):
            if ph != prev_phase:
                if prev_phase is not None:
                    ax.axvspan(start, ep - 1, alpha=0.07,
                               color=PHASE_COLORS.get(prev_phase, "#888"))
                    ax.axvline(ep - 0.5, color="#555", linestyle="--", linewidth=0.8)
                start = ep
                prev_phase = ph
        # Last segment
        ax.axvspan(start, epochs[-1], alpha=0.07,
                   color=PHASE_COLORS.get(prev_phase, "#888"))

    # ── Accuracy ─────────────────────────────────────────────────────────────
    _shade_phases(ax1)
    ax1.plot(epochs, train_acc, color="#4C9BE8", linewidth=1.8, label="Train Acc")
    ax1.plot(epochs, test_acc,  color="#5CBF7A", linewidth=1.8, linestyle="--", label="Test Acc")
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.yaxis.set_major_formatter(PercentFormatter())
    ax1.legend(fontsize=10, framealpha=0.3)
    ax1.grid(linewidth=0.4)

    # ── Loss ─────────────────────────────────────────────────────────────────
    _shade_phases(ax2)
    ax2.plot(epochs, train_loss, color="#E8914C", linewidth=1.8, label="Train Loss")
    ax2.plot(epochs, test_loss,  color="#E84C6B", linewidth=1.8, linestyle="--", label="Test Loss")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.legend(fontsize=10, framealpha=0.3)
    ax2.grid(linewidth=0.4)

    # Phase legend
    phase_patches = [
        mpatches.Patch(color=PHASE_COLORS[p], alpha=0.5, label=PHASE_LABELS[p])
        for p in PHASE_COLORS
    ]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + phase_patches, fontsize=9, framealpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Training curves → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sparsity vs. Accuracy comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(results: list[dict], out_path: str) -> None:
    """
    Bar chart comparing test accuracy and soft sparsity across all lambda runs.
    `results` is a list of dicts with keys: lambda, test_acc, soft_sparsity.
    """
    _apply_style()

    lambdas   = [str(r["lambda"])        for r in results]
    accs      = [r["test_acc"]           for r in results]
    sparsities= [r["soft_sparsity"]      for r in results]

    x = np.arange(len(lambdas))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, accs,       width, label="Test Accuracy (%)",
                    color="#4C9BE8", alpha=0.85)
    bars2 = ax2.bar(x + width/2, sparsities, width, label="Soft Sparsity (gates<0.01)",
                    color="#E84C6B", alpha=0.85)

    # Value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax1.set_xlabel("λ (Sparsity Strength)", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11, color="#4C9BE8")
    ax2.set_ylabel("Soft Sparsity (%)", fontsize=11, color="#E84C6B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(lambdas)
    ax1.set_title("Accuracy vs. Sparsity Trade-off", fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, framealpha=0.3)

    ax1.grid(axis="y", linewidth=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Comparison chart → {out_path}")
