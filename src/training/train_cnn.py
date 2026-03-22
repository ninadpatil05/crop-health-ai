"""
src/training/train_cnn.py
==========================
End-to-end training loop for the 5-class crop disease CNN.

Usage
-----
    python -m src.training.train_cnn
    # or from the project root:
    python src/training/train_cnn.py

Outputs
-------
    models/cnn/best_model.pt          – best checkpoint (by val accuracy)
    outputs/metrics/cnn_history.json  – per-epoch loss / accuracy history
    outputs/metrics/cnn_curves.png    – loss + accuracy training curves
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from src.training.cnn_model import (   # noqa: E402
    CropDiseaseCNN,
    CropDataset,
    get_dataloaders,
    build_model,
    print_model_summary,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV   = PROJECT_ROOT / "data" / "plantvillage" / "train.csv"
VAL_CSV     = PROJECT_ROOT / "data" / "plantvillage" / "val.csv"

MODEL_DIR   = PROJECT_ROOT / "models" / "cnn"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

MODEL_PATH      = MODEL_DIR   / "best_model.pt"
HISTORY_PATH    = METRICS_DIR / "cnn_history.json"
CURVES_PATH     = METRICS_DIR / "cnn_curves.png"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE  = 32
MAX_EPOCHS  = 30
PATIENCE    = 7
LR          = 1e-4
WEIGHT_DECAY = 1e-4
CLASS_NAMES = ["Healthy", "Fungal Disease", "Bacterial", "Pest Damage", "Stress"]
NUM_CLASSES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    """Create output directories if they don't exist yet."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _compute_class_weights(train_csv: Path, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to handle label imbalance.

    Formula:  w_c = N / (C * n_c)
    where N = total samples, C = number of classes, n_c = samples in class c.
    """
    df     = pd.read_csv(train_csv)
    counts = df["label"].value_counts().sort_index().values   # shape (C,)
    n_total = len(df)
    weights = n_total / (NUM_CLASSES * counts)
    t = torch.FloatTensor(weights).to(device)
    print(f"\n  Class weights : {[f'{w:.3f}' for w in weights]}")
    return t


def _run_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    phase:     str,
) -> tuple[float, float]:
    """
    Run one training or validation epoch.

    Returns
    -------
    (avg_loss, accuracy)
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += images.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def _plot_curves(history: dict, save_path: Path) -> None:
    """Save loss and accuracy curves as a side-by-side PNG."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CNN Training Curves", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, [v * 100 for v in history["train_acc"]], "b-o",
            label="Train Acc", markersize=4)
    ax.plot(epochs, [v * 100 for v in history["val_acc"]],   "r-o",
            label="Val Acc",   markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train() -> None:
    _ensure_dirs()

    # ------------------------------------------------------------------
    # 1. Device
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    print("\n  Loading datasets …")
    loaders = get_dataloaders(TRAIN_CSV, VAL_CSV, batch_size=BATCH_SIZE)
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    print(f"  Train batches : {len(train_loader)}   "
          f"Val batches : {len(val_loader)}")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = build_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
    print_model_summary(model)

    # ------------------------------------------------------------------
    # 4. Loss — class-weighted cross entropy
    # ------------------------------------------------------------------
    weights   = _compute_class_weights(TRAIN_CSV, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ------------------------------------------------------------------
    # 5. Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=True
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [],
    }

    best_acc          = 0.0
    best_epoch        = 0
    patience_counter  = 0

    header = (
        f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} "
        f"| {'Val Acc':>7} | {'LR':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 3))

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, "train"
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, "val"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(
            f"  {epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} "
            f"| {val_acc*100:>6.2f}% | {current_lr:>10.2e}  "
            f"({elapsed:.1f}s)"
        )

        # Checkpoint
        if val_acc > best_acc:
            best_acc     = val_acc
            best_epoch   = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names":      CLASS_NAMES,
                    "val_accuracy":     best_acc,
                    "epoch":            best_epoch,
                },
                MODEL_PATH,
            )
            print(f"           ✓ New best — saved checkpoint  (val_acc={best_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping triggered after {epoch} epochs "
                      f"(no improvement for {PATIENCE} epochs).")
                # Save incremental history before breaking
                with open(HISTORY_PATH, "w") as f:
                    json.dump(history, f, indent=2)
                break

        # ── Incremental history save after every epoch ──────────────────
        # Persists progress even if the process is cancelled mid-run.
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Plot curves (from whatever history was recorded)
    # ------------------------------------------------------------------
    if history["train_loss"]:
        print(f"\n  Saving history  → {HISTORY_PATH}")
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
        _plot_curves(history, CURVES_PATH)

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  Training complete")
    print("=" * 50)
    print(f"  Best epoch        : {best_epoch}")
    print(f"  Best val accuracy : {best_acc * 100:.2f}%")
    print(f"  Model saved to    : {MODEL_PATH}")
    print("=" * 50 + "\n")

    if best_acc < 0.85:
        print("  ⚠  Val accuracy below 85% target — consider more epochs,")
        print("     lower LR, or unfreezing more backbone layers.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
