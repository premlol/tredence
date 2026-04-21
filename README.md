# Self-Pruning Neural Network — CIFAR-10

A PyTorch implementation of **Learnable Gated Sparsity** for CIFAR-10 image classification. The network learns *which of its own weights to remove* during training using a 4-phase curriculum, without any manual pruning heuristics.

---

## How It Works

Every weight `W` in the classifier head is paired with a learnable **gate score** `G`. During the forward pass:

```
W_effective = W ⊙ sigmoid(gate_scores)
```

The loss function creates a tug-of-war between accuracy and parsimony:

```
Loss = CrossEntropy(ŷ, y)  +  λ · Σ|gates|
```

As `λ` pushes gates toward zero, the network **prunes its own weakest connections.**

---

## Project Structure

```
nn2 pruner/
├── main.py          ← Entry point (run this)
├── config.py        ← All hyperparameters
├── requirements.txt
│
├── src/
│   ├── model.py     ← PrunableLinear + PrunableCNN
│   ├── train.py     ← 4-phase training loop
│   ├── dataset.py   ← CIFAR-10 loader
│   ├── utils.py     ← Checkpointing, metrics
│   └── visualize.py ← Gate histogram, training curves, comparison plot
│
├── data/            ← CIFAR-10 auto-downloaded here
└── outputs/         ← All results saved here
    ├── lambda_0.0001/
    │   ├── model_checkpoint.pt
    │   ├── gate_histogram.png
    │   └── training_curves.png
    ├── lambda_0.001/  ...
    ├── lambda_0.01/   ...
    └── results_summary.csv
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU recommended.** CPU-only training works but takes ~30–60 min per lambda run.

### 2. Run the full experiment

```bash
python main.py
```

This will:
1. Auto-download CIFAR-10 to `./data/`
2. Run **3 experiments** (λ = 0.0001, 0.001, 0.01) sequentially
3. Save model checkpoints, plots, and a CSV summary to `./outputs/`

### 3. View results

| Output File | What it shows |
|---|---|
| `outputs/results_summary.csv` | λ, test accuracy, sparsity % for all runs |
| `outputs/lambda_*/gate_histogram.png` | Gate value distribution with spike at 0 |
| `outputs/lambda_*/training_curves.png` | Accuracy & loss across all 4 phases |
| `outputs/comparison_chart.png` | Accuracy vs. sparsity across all λ values |

---

## Configuring Hyperparameters

Edit `config.py` before running:

```python
LAMBDAS         = [0.0001, 0.001, 0.01]  # Sparsity strengths
WARMUP_EPOCHS   = 10   # Train without pruning pressure
SPARSIFY_EPOCHS = 15   # Ramp up lambda gradually
FINETUNE_EPOCHS = 11   # Retrain sparse network
PRUNE_THRESHOLD = 0.05 # Freeze any gate ≤ this value permanently
BATCH_SIZE      = 128
LR              = 1e-3
```

---

## The 4-Phase Training Curriculum

| Phase | Epochs | Lambda | Description |
|---|---|---|---|
| **1. Warm-up** | 0–9 | Off | Establish feature representations |
| **2. Sparsification** | 10–24 | Ramps 0→λ | Gates drift toward zero |
| **3. Hard Pruning** | — | — | Freeze gates ≤ 0.05 permanently |
| **4. Fine-tuning** | 25–35 | Off | Recover accuracy on sparse network |

---

## Key Design Decisions

- **`gate_scores` parameter**: Every `PrunableLinear` layer has a `gate_scores` tensor of the same shape as `weight`. Gates are computed as `sigmoid(gate_scores)`, keeping them in [0, 1].
- **L1 sparsity loss**: The penalty is `Σ|sigmoid(gate_scores)|`—the L1 norm of the gate values (not raw scores), exactly as required.
- **Lambda ramp**: Lambda increases linearly during sparsification to avoid the "dead gradient" problem where gates collapse too early.
- **Hard mask buffer**: Once a gate is frozen at 0, it stays frozen via a non-learnable `hard_mask` buffer (not affected by backpropagation).

---

## Expected Results

| λ | Test Accuracy | Soft Sparsity (gates < 0.01) |
|---|---|---|
| 0.0001 | ~82–85% | ~5% |
| 0.001 | ~78–82% | ~30–50% |
| 0.01 | ~65–75% | ~70–90% |

*Exact values depend on hardware, random seed, and number of epochs.*

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- matplotlib, numpy, tqdm, pandas
