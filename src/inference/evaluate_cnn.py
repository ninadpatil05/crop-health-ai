"""
src/inference/evaluate_cnn.py
==============================
Evaluate the trained 5-class crop disease CNN on the held-out test set.

Usage
-----
    python -m src.inference.evaluate_cnn
    # or from the project root:
    python src/inference/evaluate_cnn.py

Outputs
-------
    outputs/metrics/cnn_report.txt         – sklearn classification report
    outputs/metrics/confusion_matrix.png   – seaborn heatmap
    outputs/metrics/sample_predictions.png – 10 random test image predictions
    outputs/metrics/cnn_predictions.json   – per-image predictions for risk_mapper.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from src.training.cnn_model import build_model, get_transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH   = PROJECT_ROOT / "models"  / "cnn"     / "best_model.pt"
TEST_CSV     = PROJECT_ROOT / "data"    / "plantvillage" / "test.csv"
METRICS_DIR  = PROJECT_ROOT / "outputs" / "metrics"

REPORT_PATH      = METRICS_DIR / "cnn_report.txt"
CM_PATH          = METRICS_DIR / "confusion_matrix.png"
SAMPLE_PATH      = METRICS_DIR / "sample_predictions.png"
PREDICTIONS_PATH = METRICS_DIR / "cnn_predictions.json"

BATCH_SIZE = 32
DEFAULT_CLASS_NAMES = ["Healthy", "Fungal Disease", "Bacterial", "Pest Damage", "Stress"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: Path, device: torch.device):
    """Load the best checkpoint and return (model, class_names)."""
    checkpoint = torch.load(str(model_path), map_location="cpu")
    model = build_model(num_classes=5, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    model.to(device)
    model.eval()
    print(f"  Loaded checkpoint  → val_accuracy: {checkpoint.get('val_accuracy', 0)*100:.2f}%")
    print(f"  Class names        : {class_names}")
    return model, class_names


def _run_inference(
    model: torch.nn.Module,
    test_csv: Path,
    device: torch.device,
) -> tuple[list[int], list[int], list[list[float]], list[str]]:
    """
    Run model inference over all test images.

    Returns
    -------
    all_preds  : predicted class indices
    all_labels : ground-truth class indices
    all_probs  : softmax probability vectors
    all_paths  : image paths (str)
    """
    df = pd.read_csv(test_csv)
    transform = get_transforms("val")

    all_preds:  list[int]        = []
    all_labels: list[int]        = []
    all_probs:  list[list[float]] = []
    all_paths:  list[str]        = []

    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for batch_idx in range(n_batches):
            batch = df.iloc[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            tensors, labels, paths = [], [], []

            for _, row in batch.iterrows():
                p = Path(str(row["image_path"]))
                label = int(row["label"])
                try:
                    img = Image.open(str(p)).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), color=0)

                tensors.append(transform(img))
                labels.append(label)
                paths.append(str(p))

            batch_tensor = torch.stack(tensors).to(device)
            logits = model(batch_tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1).tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs.tolist())
            all_paths.extend(paths)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
                done = batch_idx + 1
                print(f"  Inference: {done}/{n_batches} batches  "
                      f"({done/n_batches*100:.0f}%)")

    return all_preds, all_labels, all_probs, all_paths


def _save_report(
    all_labels: list[int],
    all_preds:  list[int],
    class_names: list[str],
    save_path: Path,
) -> str:
    """Print and save the sklearn classification report. Returns the string."""
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)
    print(report)
    save_path.write_text(report)
    print(f"  Report saved → {save_path}")
    return report


def _plot_confusion_matrix(
    all_labels:  list[int],
    all_preds:   list[int],
    class_names: list[str],
    save_path:   Path,
) -> None:
    """Plot and save seaborn confusion-matrix heatmap."""
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Crop Disease CNN", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix  → {save_path}")


def _plot_sample_predictions(
    all_paths:   list[str],
    all_labels:  list[int],
    all_preds:   list[int],
    all_probs:   list[list[float]],
    class_names: list[str],
    save_path:   Path,
    n_samples:   int = 10,
) -> None:
    """Plot a grid of n_samples random test images with prediction titles."""
    indices = random.sample(range(len(all_paths)), min(n_samples, len(all_paths)))

    cols = 5
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).flatten()

    for ax_idx, img_idx in enumerate(indices):
        ax = axes[ax_idx]
        path  = all_paths[img_idx]
        true  = all_labels[img_idx]
        pred  = all_preds[img_idx]
        conf  = all_probs[img_idx][pred] * 100
        correct = true == pred

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=80)

        ax.imshow(img)
        ax.axis("off")

        title = f"True: {class_names[true]}\nPred: {class_names[pred]} ({conf:.1f}%)"
        color = "#2ecc71" if correct else "#e74c3c"   # green / red
        ax.set_title(title, fontsize=7.5, color=color, fontweight="bold")

    # Hide unused axes
    for ax in axes[len(indices):]:
        ax.axis("off")

    correct_patch = mpatches.Patch(color="#2ecc71", label="Correct")
    wrong_patch   = mpatches.Patch(color="#e74c3c", label="Wrong")
    fig.legend(handles=[correct_patch, wrong_patch],
               loc="lower right", fontsize=9, framealpha=0.8)

    fig.suptitle("Sample Test Predictions — Crop Disease CNN",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sample predictions → {save_path}")


def _save_predictions_json(
    all_paths:   list[str],
    all_preds:   list[int],
    all_probs:   list[list[float]],
    class_names: list[str],
    save_path:   Path,
) -> None:
    """Save per-image predictions as JSON for downstream risk_mapper.py."""
    out: dict[str, dict] = {}
    for path, pred, probs in zip(all_paths, all_preds, all_probs):
        out[path] = {
            "class":      class_names[pred],
            "confidence": round(float(probs[pred]), 6),
            "label":      pred,
        }
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Predictions JSON   → {save_path}  ({len(out)} images)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

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
    # 2. Load model
    # ------------------------------------------------------------------
    print(f"\n  Loading model from {MODEL_PATH} …")
    model, class_names = _load_model(MODEL_PATH, device)

    # ------------------------------------------------------------------
    # 3. Inference
    # ------------------------------------------------------------------
    print(f"\n  Running inference on {TEST_CSV} …")
    all_preds, all_labels, all_probs, all_paths = _run_inference(
        model, TEST_CSV, device
    )

    overall_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n  Test samples : {len(all_labels)}")
    print(f"  Overall acc  : {overall_acc*100:.2f}%")

    # ------------------------------------------------------------------
    # 4. Classification report
    # ------------------------------------------------------------------
    _save_report(all_labels, all_preds, class_names, REPORT_PATH)

    # ------------------------------------------------------------------
    # 5. Confusion matrix
    # ------------------------------------------------------------------
    _plot_confusion_matrix(all_labels, all_preds, class_names, CM_PATH)

    # ------------------------------------------------------------------
    # 6. Sample predictions grid
    # ------------------------------------------------------------------
    _plot_sample_predictions(
        all_paths, all_labels, all_preds, all_probs, class_names, SAMPLE_PATH
    )

    # ------------------------------------------------------------------
    # 7. Predictions JSON
    # ------------------------------------------------------------------
    _save_predictions_json(all_paths, all_preds, all_probs, class_names, PREDICTIONS_PATH)

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  Evaluation complete")
    print("=" * 50)
    print(f"  Test accuracy  : {overall_acc*100:.2f}%")
    if overall_acc >= 0.85:
        print("  ✓ Target >85% achieved!")
    else:
        print("  ⚠  Below 85% target — check training or data.")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()
