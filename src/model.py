"""
src/model.py
------------
Defines:
  - PrunableLinear : custom linear layer with learnable gate_scores
  - PrunableCNN    : CNN backbone + prunable classifier head for CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# PrunableLinear — the core building block  ✅ Checklist: uses "gate_scores"
# ══════════════════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that associates a learnable gate scalar
    with every weight via:

        W_effective = weight  ⊙  sigmoid(gate_scores)

    The gate values are kept in [0, 1] by the sigmoid. A sparsity loss term
    (L1 norm of the gates) encourages them toward exactly 0, effectively
    pruning the corresponding weight connections.

    Parameters
    ----------
    in_features  : int
    out_features : int
    bias         : bool  (default True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        # Standard learnable weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        # Bias (optional)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # ✅ Requirement: gate_scores — same shape as weight, initialised high
        #    so all gates start near sigmoid(2) ≈ 0.88 (mostly open).
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))

        # Mask for hard-pruned gates (bool tensor, not a Parameter)
        self.register_buffer("hard_mask", torch.ones(out_features, in_features, dtype=torch.bool))

    # ─── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert scores → gates in [0, 1]
        gates = torch.sigmoid(self.gate_scores)

        # Respect hard-pruned positions (frozen at 0)
        gates = gates * self.hard_mask.float()

        # Effective weight = weight ⊙ gates
        pruned_weight = self.weight * gates

        return F.linear(x, pruned_weight, self.bias)

    # ─── Gate helpers ─────────────────────────────────────────────────────────

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (sigmoid of scores), respecting hard mask."""
        gates = torch.sigmoid(self.gate_scores)
        return gates * self.hard_mask.float()

    def get_sparsity_loss(self) -> torch.Tensor:
        """✅ Requirement: L1 norm of gate values (not raw scores)."""
        gates = self.get_gates()
        return gates.abs().sum()

    def apply_hard_pruning(self, threshold: float = 0.05) -> int:
        """
        Permanently zero-out and freeze any gate whose value is <= threshold.
        Returns the count of newly pruned connections.
        """
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            newly_pruned = (gates <= threshold) & self.hard_mask
            self.hard_mask[newly_pruned] = False
            # Drive score deep negative so sigmoid ≈ 0 numerically
            self.gate_scores[~self.hard_mask] = -10.0
        return newly_pruned.sum().item()

    def sparsity_stats(self) -> dict:
        """Return a dict with total weights, active weights, and sparsity %."""
        total   = self.hard_mask.numel()
        active  = self.hard_mask.sum().item()
        pruned  = total - active
        return {
            "total":    total,
            "active":   active,
            "pruned":   pruned,
            "sparsity": pruned / total * 100,
        }

    def extra_repr(self) -> str:
        stats = self.sparsity_stats()
        return (
            f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, "
            f"sparsity={stats['sparsity']:.1f}%"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CNN Backbone blocks
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════════════
# PrunableCNN — full model
# ══════════════════════════════════════════════════════════════════════════════

class PrunableCNN(nn.Module):
    """
    3-block CNN feature extractor + prunable fully-connected classifier head.

    Architecture
    ────────────
    Input: 3×32×32 (CIFAR-10)

    Feature extractor (conv blocks, NOT pruned):
        Block 1: Conv(3→64)   + BN + ReLU + Pool  →  64×16×16
        Block 2: Conv(64→128) + BN + ReLU + Pool  →  128×8×8
        Block 3: Conv(128→256)+ BN + ReLU + Pool  →  256×4×4

    Classifier head (PrunableLinear, PRUNED):
        Flatten → 4096
        PrunableLinear(4096 → 512) + ReLU + Dropout
        PrunableLinear(512  → 256) + ReLU
        PrunableLinear(256  → 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        # ── Conv feature extractor ───────────────────────────────────────────
        self.features = nn.Sequential(
            ConvBlock(3,   64,  pool=True),
            ConvBlock(64,  128, pool=True),
            ConvBlock(128, 256, pool=True),
        )

        # Adaptive pool so the code is resolution-independent
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── Prunable classifier head ─────────────────────────────────────────
        self.fc1      = PrunableLinear(256 * 4 * 4, 512)
        self.relu1    = nn.ReLU(inplace=True)
        self.dropout  = nn.Dropout(p=dropout)

        self.fc2      = PrunableLinear(512, 256)
        self.relu2    = nn.ReLU(inplace=True)

        self.fc3      = PrunableLinear(256, num_classes)

        self._prunable_layers = [self.fc1, self.fc2, self.fc3]

    # ─── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(self.relu1(self.fc1(x)))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    # ─── Sparsity helpers ─────────────────────────────────────────────────────

    def get_sparsity_loss(self) -> torch.Tensor:
        """✅ Requirement: L1 norm summed across ALL prunable layers."""
        return sum(layer.get_sparsity_loss() for layer in self._prunable_layers)

    def get_all_gates(self) -> torch.Tensor:
        """Concatenate all gate values into a single flat tensor (for plotting)."""
        return torch.cat([layer.get_gates().detach().cpu().flatten()
                          for layer in self._prunable_layers])

    def apply_hard_pruning(self, threshold: float = 0.05) -> int:
        """Apply hard pruning to all prunable layers. Returns total pruned count."""
        total_pruned = 0
        for layer in self._prunable_layers:
            total_pruned += layer.apply_hard_pruning(threshold)
        return total_pruned

    def global_sparsity(self) -> float:
        """Return the overall percentage of permanently pruned weights."""
        total  = sum(l.sparsity_stats()["total"]  for l in self._prunable_layers)
        pruned = sum(l.sparsity_stats()["pruned"] for l in self._prunable_layers)
        return pruned / total * 100 if total > 0 else 0.0

    def parameter_count(self) -> dict:
        """Return total and active (non-pruned) parameter counts."""
        total  = sum(p.numel() for p in self.parameters())
        pruned = sum(l.sparsity_stats()["pruned"] for l in self._prunable_layers)
        return {"total": total, "active": total - pruned, "pruned": pruned}
