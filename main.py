"""
main.py
-------
Entry point for the Self-Pruning Neural Network experiment.

Runs three independent experiments (one per lambda value defined in config.py),
saves per-run outputs to ./outputs/lambda_<lam>/, and writes a results summary
to ./outputs/results_summary.csv.

Usage
-----
    python main.py

All hyperparameters live in config.py -- no command-line args needed.
"""

import os
import sys
import torch

import config
from src.dataset   import get_cifar10_loaders
from src.model     import PrunableCNN
from src.train     import train
from src.utils     import save_checkpoint, append_result_row, compute_soft_sparsity
from src.visualize import plot_gate_histogram, plot_training_curves, plot_comparison


# ==============================================================================
# Startup banner
# ==============================================================================

def _startup_banner():
    print("=" * 64)
    print("  Self-Pruning Neural Network  --  CIFAR-10")
    print("=" * 64)
    print(f"  Device         : {config.DEVICE.upper()}")
    print(f"  Lambdas        : {config.LAMBDAS}")
    print(f"  Epochs         : warmup={config.WARMUP_EPOCHS} | "
          f"sparsify={config.SPARSIFY_EPOCHS} | "
          f"finetune={config.FINETUNE_EPOCHS}")
    print(f"  Batch size     : {config.BATCH_SIZE}")
    print(f"  Prune threshold: {config.PRUNE_THRESHOLD}")
    if config.DEVICE == "cpu":
        print("\n  [!] WARNING: No GPU detected. Training on CPU will be slow.")
        print("      Consider running on a CUDA-enabled machine for faster results.")
    print("=" * 64 + "\n")


# ==============================================================================
# Single experiment
# ==============================================================================

def run_experiment(lam, train_loader, test_loader, run_dir):
    """
    Train one self-pruning model for the given lambda.
    Returns a result summary dict for the CSV.
    """
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n>>> Starting experiment  lambda = {lam} <<<\n")

    model = PrunableCNN(num_classes=config.NUM_CLASSES)
    param_info = model.parameter_count()
    print(f"  Model parameters: {param_info['total']:,}")

    history = train(
        model           = model,
        train_loader    = train_loader,
        test_loader     = test_loader,
        device          = config.DEVICE,
        target_lambda   = lam,
        warmup_epochs   = config.WARMUP_EPOCHS,
        sparsify_epochs = config.SPARSIFY_EPOCHS,
        finetune_epochs = config.FINETUNE_EPOCHS,
        prune_threshold = config.PRUNE_THRESHOLD,
        lr              = config.LR,
        weight_decay    = config.WEIGHT_DECAY,
    )

    # -- Final metrics ---------------------------------------------------------
    final           = history[-1]
    soft_sparsity   = compute_soft_sparsity(model, threshold=0.01)
    hard_sparsity   = model.global_sparsity()
    param_info_post = model.parameter_count()

    print(f"\n  -- Final results for lambda={lam} --")
    print(f"     Test accuracy  : {final['test_acc']:.2f}%")
    print(f"     Soft sparsity  : {soft_sparsity:.2f}%  (gates < 0.01)")
    print(f"     Hard sparsity  : {hard_sparsity:.2f}%  (permanently pruned)")
    print(f"     Active params  : {param_info_post['active']:,} / {param_info_post['total']:,}")

    # -- Save checkpoint -------------------------------------------------------
    ckpt_path = os.path.join(run_dir, "model_checkpoint.pt")
    save_checkpoint(
        {"model_state": model.state_dict(), "history": history, "lambda": lam},
        ckpt_path,
    )
    print(f"  Checkpoint saved -> {ckpt_path}")

    # -- Plots -----------------------------------------------------------------
    plot_gate_histogram(
        model    = model,
        out_path = os.path.join(run_dir, "gate_histogram.png"),
        lam      = lam,
    )
    plot_training_curves(
        history  = history,
        out_path = os.path.join(run_dir, "training_curves.png"),
        lam      = lam,
    )

    return {
        "lambda":        lam,
        "test_acc":      round(final["test_acc"], 3),
        "soft_sparsity": round(soft_sparsity, 3),
        "hard_sparsity": round(hard_sparsity, 3),
        "active_params": param_info_post["active"],
        "total_params":  param_info_post["total"],
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    _startup_banner()

    # -- Data ------------------------------------------------------------------
    print("Loading CIFAR-10 dataset ...")
    train_loader, test_loader, class_names = get_cifar10_loaders(
        data_dir    = config.DATA_DIR,
        batch_size  = config.BATCH_SIZE,
        num_workers = config.NUM_WORKERS,
    )
    print(f"  Classes: {', '.join(class_names)}\n")

    # -- Run experiments -------------------------------------------------------
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(config.OUTPUT_DIR, "results_summary.csv")

    # Remove stale CSV if re-running
    if os.path.exists(csv_path):
        os.remove(csv_path)

    all_results = []
    for lam in config.LAMBDAS:
        run_dir = os.path.join(config.OUTPUT_DIR, f"lambda_{lam}")
        result  = run_experiment(lam, train_loader, test_loader, run_dir)
        all_results.append(result)
        append_result_row(csv_path, result)

    # -- Comparison plot -------------------------------------------------------
    plot_comparison(
        results  = all_results,
        out_path = os.path.join(config.OUTPUT_DIR, "comparison_chart.png"),
    )

    # -- Summary table ---------------------------------------------------------
    sep = "=" * 64
    print(f"\n{sep}")
    print("  RESULTS SUMMARY")
    print(sep)
    print(f"  {'Lambda':>10}  {'Test Acc':>10}  {'Soft Sparsity':>15}  {'Hard Sparsity':>15}")
    print("  " + "-" * 56)
    for r in all_results:
        print(
            f"  {r['lambda']:>10}  "
            f"{r['test_acc']:>9.2f}%  "
            f"{r['soft_sparsity']:>14.2f}%  "
            f"{r['hard_sparsity']:>14.2f}%"
        )
    print(sep)
    print(f"\n  Full results saved to -> {csv_path}")
    print("  All plots saved under -> ./outputs/\n")


if __name__ == "__main__":
    main()
