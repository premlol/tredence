"""
src/train.py
------------
Implements the 4-phase Self-Pruning training curriculum:

  Phase 1  Warm-up        Train without sparsity pressure (baseline formation)
  Phase 2  Sparsification Gradually ramp lambda; gates drift toward 0
  Phase 3  Hard Pruning   Freeze any gate <= threshold permanently (single pass)
  Phase 4  Fine-tuning    Retrain sparse network to recover accuracy
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from src.model import PrunableCNN
from src.utils import compute_soft_sparsity, print_phase_header


# ==============================================================================
# Per-epoch helpers
# ==============================================================================

def _train_epoch(model, loader, optimizer, criterion, device, lam=0.0):
    """
    One training epoch.

    Loss = CrossEntropy(y, y_hat) + lam * sum|gates|   (L1 sparsity term)

    Returns
    -------
    avg_loss : float
    accuracy : float  (0-100)
    """
    model.train()
    total_loss = correct = seen = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # -- Classification loss
        ce_loss = criterion(logits, labels)

        # -- Sparsity loss (L1 norm of all gate values)
        sparsity_loss = model.get_sparsity_loss() if lam > 0 else torch.tensor(0.0)
        loss = ce_loss + lam * sparsity_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        seen       += images.size(0)

    return total_loss / seen, correct / seen * 100


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    """Evaluate on test set. Returns avg_loss, accuracy."""
    model.eval()
    total_loss = correct = seen = 0

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        seen       += images.size(0)

    return total_loss / seen, correct / seen * 100


# ==============================================================================
# Main training function
# ==============================================================================

def train(
    model,
    train_loader,
    test_loader,
    device,
    target_lambda,
    warmup_epochs,
    sparsify_epochs,
    finetune_epochs,
    prune_threshold,
    lr,
    weight_decay,
    on_epoch_end=None,
):
    """
    Run the full 4-phase self-pruning curriculum.

    on_epoch_end : optional callback called at end of each epoch with a metrics dict.

    Returns
    -------
    history : list of per-epoch metric dicts
    """

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing LR over full training duration
    total_epochs = warmup_epochs + sparsify_epochs + finetune_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    history = []
    global_epoch = 0

    # --------------------------------------------------------------------------
    # Phase 1: Warm-up
    # --------------------------------------------------------------------------
    print_phase_header(f"Phase 1 -- Warm-up  ({warmup_epochs} epochs, lambda=0)")

    for ep in range(warmup_epochs):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device, lam=0.0)
        te_loss, te_acc = _eval_epoch(model, test_loader, criterion, device)
        scheduler.step()
        sparsity = compute_soft_sparsity(model)

        row = {
            "phase": "warmup", "epoch": global_epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "test_loss": te_loss,  "test_acc": te_acc,
            "lambda": 0.0, "soft_sparsity": sparsity,
        }
        history.append(row)
        if on_epoch_end:
            on_epoch_end(row)

        print(
            f"  Epoch {global_epoch:3d} | "
            f"Train {tr_acc:.1f}% | Test {te_acc:.1f}% | "
            f"Sparsity {sparsity:.1f}%"
        )
        global_epoch += 1

    # --------------------------------------------------------------------------
    # Phase 2: Sparsification (lambda ramps 0 -> target_lambda linearly)
    # --------------------------------------------------------------------------
    print_phase_header(
        f"Phase 2 -- Sparsification  ({sparsify_epochs} epochs, lambda -> {target_lambda})"
    )

    for ep in range(sparsify_epochs):
        # Linear ramp: avoids killing gates too early in the phase
        ramp = (ep + 1) / sparsify_epochs
        current_lam = target_lambda * ramp

        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device, lam=current_lam)
        te_loss, te_acc = _eval_epoch(model, test_loader, criterion, device)
        scheduler.step()
        sparsity = compute_soft_sparsity(model)

        row = {
            "phase": "sparsify", "epoch": global_epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "test_loss": te_loss,  "test_acc": te_acc,
            "lambda": current_lam, "soft_sparsity": sparsity,
        }
        history.append(row)
        if on_epoch_end:
            on_epoch_end(row)

        print(
            f"  Epoch {global_epoch:3d} | "
            f"Train {tr_acc:.1f}% | Test {te_acc:.1f}% | "
            f"lam={current_lam:.5f} | Sparsity {sparsity:.1f}%"
        )
        global_epoch += 1

    # --------------------------------------------------------------------------
    # Phase 3: Hard Pruning (freeze gates <= threshold permanently)
    # --------------------------------------------------------------------------
    print_phase_header(f"Phase 3 -- Hard Pruning  (threshold={prune_threshold})")

    pruned_count  = model.apply_hard_pruning(threshold=prune_threshold)
    hard_sparsity = model.global_sparsity()
    param_info    = model.parameter_count()

    print(f"  Connections pruned this step : {pruned_count:,}")
    print(f"  Overall sparsity (hard mask) : {hard_sparsity:.1f}%")
    print(f"  Active / Total params        : {param_info['active']:,} / {param_info['total']:,}")

    # --------------------------------------------------------------------------
    # Phase 4: Fine-tuning (lower LR, no sparsity pressure)
    # --------------------------------------------------------------------------
    print_phase_header(f"Phase 4 -- Fine-tuning  ({finetune_epochs} epochs, lambda=0)")

    # Reduce LR for fine-tuning
    for pg in optimizer.param_groups:
        pg["lr"] = lr * 0.1

    for ep in range(finetune_epochs):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device, lam=0.0)
        te_loss, te_acc = _eval_epoch(model, test_loader, criterion, device)
        sparsity = compute_soft_sparsity(model)

        row = {
            "phase": "finetune", "epoch": global_epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "test_loss": te_loss,  "test_acc": te_acc,
            "lambda": 0.0, "soft_sparsity": sparsity,
        }
        history.append(row)
        if on_epoch_end:
            on_epoch_end(row)

        print(
            f"  Epoch {global_epoch:3d} | "
            f"Train {tr_acc:.1f}% | Test {te_acc:.1f}% | "
            f"Sparsity {sparsity:.1f}%"
        )
        global_epoch += 1

    return history
