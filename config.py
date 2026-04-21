"""
config.py
---------
Central configuration for the Self-Pruning Neural Network experiment.
Modify values here to control training behaviour without touching source code.
"""

import torch

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Data ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "./data"          # CIFAR-10 will be auto-downloaded here
NUM_WORKERS = 2                 # DataLoader workers (set 0 on Windows if errors)

# ─── Model ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 10                # CIFAR-10

# ─── Training Phases ──────────────────────────────────────────────────────────
# Total = WARMUP + SPARSIFY + FINETUNE epochs (hard-prune is a single event)
WARMUP_EPOCHS    = 10           # Phase 1: train without sparsity pressure
SPARSIFY_EPOCHS  = 15           # Phase 2: gradually ramp lambda
PRUNE_THRESHOLD  = 0.05         # Phase 3: freeze gates <= this value permanently
FINETUNE_EPOCHS  = 11           # Phase 4: retrain the sparse network

# ─── Optimiser ────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
LR         = 1e-3
WEIGHT_DECAY = 1e-4

# ─── Sparsity Lambdas to Benchmark ────────────────────────────────────────────
# Each value runs as a separate experiment so you can compare accuracy/sparsity.
LAMBDAS = [0.0001, 0.001, 0.01]

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./outputs"        # Per-lambda sub-dirs created automatically
