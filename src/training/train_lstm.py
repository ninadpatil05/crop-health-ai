"""
src/training/train_lstm.py
============================
End-to-end training loop for the NDVIForecaster LSTM.

Usage
-----
    python -m src.training.train_lstm

Outputs
-------
    models/lstm/best_lstm.pt              – best checkpoint (lowest val RMSE)
    outputs/metrics/lstm_predictions.png  – actual vs predicted plot (5 zones)
"""

from __future__ import annotations

import os
# Fix Windows OpenMP DLL conflict (libiomp5md.dll vs libomp.dll)
# that can crash the process before saving outputs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.lstm_model import NDVIForecaster

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
DATA_PATH      = PROJECT_ROOT / "data"    / "sentinel2" / "timeseries.npz"
MODEL_DIR      = PROJECT_ROOT / "models"  / "lstm"
MODEL_PATH     = MODEL_DIR    / "best_lstm.pt"
METRICS_DIR    = PROJECT_ROOT / "outputs" / "metrics"
PLOT_PATH      = METRICS_DIR  / "lstm_predictions.png"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
MAX_EPOCHS  = 100
PATIENCE    = 10
BATCH_SIZE  = 64
LR          = 1e-3
TRAIN_SPLIT = 0.8
WINDOW_SIZE = 8
N_ZONES     = 100
PLOT_ZONES  = [0, 10, 25, 50, 75]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return float(torch.sqrt(nn.functional.mse_loss(pred, target)).item())


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-8)


def _plot_predictions(
    model:     nn.Module,
    zone_ndvi: np.ndarray,
    device:    torch.device,
    save_path: Path,
    plot_zones: list[int] = PLOT_ZONES,
    window:    int = WINDOW_SIZE,
) -> None:
    """
    Roll the trained model over each selected zone's full series and
    overlay actual vs predicted NDVI.

    Parameters
    ----------
    model      : Trained NDVIForecaster in eval mode.
    zone_ndvi  : ndarray of shape (n_zones, T) — full NDVI series per zone.
    device     : Torch device.
    save_path  : Destination PNG path.
    plot_zones : Zone indices to plot.
    window     : Input sequence length used during training.
    """
    model.eval()
    fig, axes = plt.subplots(len(plot_zones), 1,
                             figsize=(12, 3 * len(plot_zones)),
                             sharex=False)

    if len(plot_zones) == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, zone in zip(axes, plot_zones):
            series = zone_ndvi[zone]       # shape (T,)
            T      = len(series)
            actuals, preds = [], []

            for i in range(T - window - 1):
                x_in = torch.FloatTensor(
                    series[i : i + window].reshape(1, window, 1)
                ).to(device)
                y_hat = model(x_in).cpu().numpy().flatten()

                # Record only the first predicted step for a rolling forecast
                actuals.append(series[i + window])
                preds.append(float(y_hat[0]))

            t_axis = np.arange(window, window + len(actuals))
            ax.plot(t_axis, actuals, "b-",  linewidth=1.5, label="Actual NDVI")
            ax.plot(t_axis, preds,   "r--", linewidth=1.5, label="Predicted t+1")
            ax.fill_between(t_axis,
                            [a - 0.05 for a in actuals],
                            [a + 0.05 for a in actuals],
                            alpha=0.1, color="blue")
            ax.set_title(f"Zone {zone}", fontsize=10, fontweight="bold")
            ax.set_ylabel("NDVI", fontsize=9)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Week", fontsize=10)
    fig.suptitle("LSTM NDVI Forecast — Actual vs Predicted",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
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
    # 2. Load data
    # ------------------------------------------------------------------
    print(f"\n  Loading data from {DATA_PATH} …")
    data = np.load(str(DATA_PATH))
    X         = torch.FloatTensor(data["X_sequences"])   # (N, 8, 1)
    y         = torch.FloatTensor(data["y_targets"])     # (N, 2)
    zone_ndvi = data["zone_ndvi"]                        # (100, T)

    print(f"  X shape : {tuple(X.shape)}")
    print(f"  y shape : {tuple(y.shape)}")

    # ------------------------------------------------------------------
    # 3. Train / val split (80/20, no shuffle — preserves temporal order)
    # ------------------------------------------------------------------
    n     = len(X)
    split = int(TRAIN_SPLIT * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"  Train : {len(X_train)}   Val : {len(X_val)}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # ------------------------------------------------------------------
    # 4. Model, loss, optimiser
    # ------------------------------------------------------------------
    model     = NDVIForecaster(input_size=1, hidden=64, layers=2, dropout=0.2)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params : {total_params:,}")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    best_rmse        = float("inf")
    best_epoch       = 0
    patience_counter = 0
    history          = {"train_loss": [], "val_loss": [], "val_rmse": []}

    header = (f"\n  {'Epoch':>5} | {'Train Loss':>10} | "
              f"{'Val Loss':>9} | {'Val RMSE':>9}")
    print(header)
    print("  " + "-" * (len(header) - 3))

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
        train_loss = train_loss_sum / len(X_train)

        # ── Validation ─────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * len(xb)
                all_preds.append(pred.cpu())
                all_targets.append(yb.cpu())

        val_loss  = val_loss_sum / len(X_val)
        val_preds = torch.cat(all_preds)
        val_tgt   = torch.cat(all_targets)
        val_rmse  = _rmse(val_preds, val_tgt)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)

        elapsed = time.time() - t0
        print(f"  {epoch:>5} | {train_loss:>10.6f} | "
              f"{val_loss:>9.6f} | {val_rmse:>9.6f}  ({elapsed:.1f}s)")

        # ── Checkpoint ─────────────────────────────────────────────────
        if val_rmse < best_rmse:
            best_rmse    = val_rmse
            best_epoch   = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size":       1,
                    "val_rmse":         best_rmse,
                    "epoch":            best_epoch,
                },
                MODEL_PATH,
            )
            print(f"           ✓ Best RMSE = {best_rmse:.6f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping after epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs).")
                break

    # ------------------------------------------------------------------
    # 6. Load best model and compute final metrics
    # ------------------------------------------------------------------
    ckpt = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        final_preds = model(X_val.to(device)).cpu().numpy()
    final_targets = y_val.numpy()

    final_rmse = float(np.sqrt(np.mean((final_preds - final_targets) ** 2)))
    final_r2   = _r2(final_preds, final_targets)

    # ------------------------------------------------------------------
    # 7. Plot actual vs predicted
    # ------------------------------------------------------------------
    print(f"\n  Plotting actual vs predicted for zones {PLOT_ZONES} …")
    _plot_predictions(model, zone_ndvi, device, PLOT_PATH)

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  LSTM Training Complete")
    print("=" * 50)
    print(f"  Best epoch  : {best_epoch}")
    print(f"  Val RMSE    : {final_rmse:.6f}  (target < 0.05)")
    print(f"  Val R²      : {final_r2:.6f}  (target > 0.85)")
    if final_rmse < 0.05:
        print("  ✓ RMSE target achieved!")
    else:
        print("  ⚠  RMSE above 0.05 — consider more epochs or hidden units.")
    if final_r2 > 0.85:
        print("  ✓ R² target achieved!")
    else:
        print("  ⚠  R² below 0.85 — model may need tuning.")
    print(f"  Model saved : {MODEL_PATH}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
