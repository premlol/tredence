"""
src/utils.py
------------
Helper utilities: checkpoint save/load, metric logging, sparsity reporting.
"""

import os
import csv
import torch


# -- Checkpoint helpers --------------------------------------------------------

def save_checkpoint(state, path):
    """Save training state (model weights, epoch, metrics) to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    """Load a checkpoint and restore model (+ optionally optimizer) weights."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt


# -- Results CSV ---------------------------------------------------------------

def append_result_row(csv_path, row):
    """Append a single result row to the summary CSV (creates headers if new)."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -- Sparsity reporting --------------------------------------------------------

def compute_soft_sparsity(model, threshold=0.01):
    """
    Fraction of gate values (sigmoid of gate_scores) below `threshold`.
    This is the metric required by the case study (threshold = 1e-2).
    """
    from src.model import PrunableLinear
    all_gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            all_gates.append(m.get_gates().detach().cpu().flatten())
    if not all_gates:
        return 0.0
    gates = torch.cat(all_gates)
    return (gates < threshold).float().mean().item() * 100


def print_phase_header(phase_name):
    width = 60
    print("\n" + "=" * width)
    print(f"  {phase_name.upper()}")
    print("=" * width)
