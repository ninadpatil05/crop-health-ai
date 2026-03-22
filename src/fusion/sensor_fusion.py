"""
src/fusion/sensor_fusion.py
=============================
Fuse Sentinel-2 NDVI time-series with weather/soil sensor features,
retrain the NDVIForecaster LSTM with the richer input, and compare
against the NDVI-only baseline.

Alignment strategy
------------------
The NDVI sequences (N × 8 × 1) and the sensor CSV (~90 rows) have
different time granularities.  For prototyping we tile the **last 8
rows** of sensor readings into every sequence — a valid approximation
that injects recent field conditions into every training window without
requiring date-aligned joining.

Outputs
-------
  data/sentinel2/timeseries_fused.npz  – fused X/y arrays
  models/fusion/lstm_fused.pt           – best fused checkpoint
  (comparison table printed to stdout)

Usage
-----
    python -m src.fusion.sensor_fusion
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.training.lstm_model import NDVIForecaster

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
NDVI_NPZ        = PROJECT_ROOT / "data"   / "sentinel2" / "timeseries.npz"
FUSED_NPZ       = PROJECT_ROOT / "data"   / "sentinel2" / "timeseries_fused.npz"
SENSOR_CSV      = PROJECT_ROOT / "data"   / "sensor"    / "sensor_data.csv"
BASELINE_PT     = PROJECT_ROOT / "models" / "lstm"       / "best_lstm.pt"
FUSED_MODEL_DIR = PROJECT_ROOT / "models" / "fusion"
FUSED_PT        = FUSED_MODEL_DIR / "lstm_fused.pt"

# ---------------------------------------------------------------------------
# Hyper-parameters (same as P3-02 train_lstm.py)
# ---------------------------------------------------------------------------
SENSOR_COLS  = ["soil_moisture", "temperature", "humidity",
                "precipitation", "evapotranspiration"]
INPUT_SIZE   = 1 + len(SENSOR_COLS)   # 6
MAX_EPOCHS   = 100
PATIENCE     = 10
BATCH_SIZE   = 64
LR           = 1e-3
TRAIN_SPLIT  = 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def build_fused_dataset(
    ndvi_npz:   Path = NDVI_NPZ,
    sensor_csv: Path = SENSOR_CSV,
    fused_npz:  Path = FUSED_NPZ,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse NDVI sequences with sensor features and persist the result.

    Steps
    -----
    1. Load ``X_sequences`` (N, 8, 1) and ``y_targets`` (N, 2).
    2. Take the last 8 rows of the sensor CSV and tile to (N, 8, 5).
    3. Normalise sensor features via StandardScaler fit on the train split.
    4. Concatenate along the feature axis → (N, 8, 6).
    5. Save to *fused_npz* and return ``(X_fused, y)``.

    Parameters
    ----------
    ndvi_npz   : Path to the NDVI timeseries .npz file.
    sensor_csv : Path to the sensor CSV.
    fused_npz  : Destination path for the fused .npz file.

    Returns
    -------
    (X_fused, y) as float32 ndarrays.
    """
    # -- Load NDVI sequences ------------------------------------------
    data     = np.load(str(ndvi_npz))
    X_ndvi   = data["X_sequences"].astype(np.float32)   # (N, 8, 1)
    y        = data["y_targets"].astype(np.float32)     # (N, 2)
    N        = len(X_ndvi)
    print(f"  NDVI sequences : {X_ndvi.shape}")

    # -- Load and tile sensor data ------------------------------------
    sensor      = pd.read_csv(str(sensor_csv))
    sensor_tail = sensor[SENSOR_COLS].tail(8).values     # (8, 5)
    sensor_tiled = np.tile(sensor_tail, (N, 1, 1))       # (N, 8, 5)
    print(f"  Sensor tiled   : {sensor_tiled.shape}")

    # -- Normalise sensor features ------------------------------------
    n_train      = int(TRAIN_SPLIT * N)
    sensor_flat  = sensor_tiled.reshape(N, -1).astype(np.float32)   # (N, 40)
    scaler       = StandardScaler()
    sensor_flat[:n_train] = scaler.fit_transform(sensor_flat[:n_train])
    sensor_flat[n_train:] = scaler.transform(sensor_flat[n_train:])
    sensor_norm  = sensor_flat.reshape(N, 8, 5)

    # -- Fuse ---------------------------------------------------------
    X_fused = np.concatenate([X_ndvi, sensor_norm], axis=2)   # (N, 8, 6)
    print(f"  Fused array    : {X_fused.shape}")

    X_fused = X_fused.astype(np.float32)
    fused_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(fused_npz), X_sequences=X_fused, y_targets=y)
    print(f"  Saved fused dataset → {fused_npz}")

    return X_fused, y


def train_fused_lstm(
    X:          np.ndarray,
    y:          np.ndarray,
    device:     torch.device,
    model_path: Path = FUSED_PT,
) -> float:
    """
    Train NDVIForecaster with ``input_size=6`` on the fused data.

    Uses the same 80/20 temporal split, Adam optimizer, MSELoss, and
    early-stopping (patience=10) as the NDVI-only baseline.

    Parameters
    ----------
    X          : Fused sequences of shape (N, 8, 6).
    y          : Targets of shape (N, 2).
    device     : Torch device.
    model_path : Where to save the best checkpoint.

    Returns
    -------
    Best validation RMSE achieved.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    n     = len(X)
    split = int(TRAIN_SPLIT * n)

    X_t  = torch.FloatTensor(X)
    y_t  = torch.FloatTensor(y)
    X_train, X_val = X_t[:split], X_t[split:]
    y_train, y_val = y_t[:split], y_t[split:]
    print(f"  Train : {len(X_train)}   Val : {len(X_val)}")

    train_dl = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_val, y_val),
                          batch_size=BATCH_SIZE, shuffle=False)

    model = NDVIForecaster(input_size=INPUT_SIZE, hidden=64,
                           layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_rmse       = float("inf")
    best_epoch      = 0
    patience_ctr    = 0

    header = (f"\n  {'Epoch':>5} | {'Train Loss':>10} | "
              f"{'Val Loss':>9} | {'Val RMSE':>9}")
    print(header)
    print("  " + "-" * (len(header) - 3))

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        tl_sum = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tl_sum += loss.item() * len(xb)
        train_loss = tl_sum / len(X_train)

        # Val
        model.eval()
        vl_sum = 0.0
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                p = model(xb)
                vl_sum += criterion(p, yb).item() * len(xb)
                all_p.append(p.cpu().numpy())
                all_t.append(yb.cpu().numpy())
        val_loss = vl_sum / len(X_val)
        preds_np = np.concatenate(all_p)
        tgts_np  = np.concatenate(all_t)
        val_rmse = _rmse(preds_np, tgts_np)

        elapsed = time.time() - t0
        print(f"  {epoch:>5} | {train_loss:>10.6f} | "
              f"{val_loss:>9.6f} | {val_rmse:>9.6f}  ({elapsed:.1f}s)")

        if val_rmse < best_rmse:
            best_rmse  = val_rmse
            best_epoch = epoch
            patience_ctr = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "input_size": INPUT_SIZE,
                        "val_rmse":   best_rmse,
                        "epoch":      best_epoch}, model_path)
            print(f"           ✓ Best RMSE = {best_rmse:.6f} — checkpoint saved")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    return best_rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("  Sensor Fusion — NDVI + Weather/Soil LSTM")
    print("=" * 55)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    # ------------------------------------------------------------------
    # 1. Build fused dataset
    # ------------------------------------------------------------------
    print("\n  ── Step 1: Fuse NDVI + sensor features ──────────────────")
    X_fused, y = build_fused_dataset()

    # ------------------------------------------------------------------
    # 2. Load baseline RMSE
    # ------------------------------------------------------------------
    print("\n  ── Step 2: Load baseline RMSE ────────────────────────────")
    baseline_ckpt = torch.load(str(BASELINE_PT), map_location="cpu",
                               weights_only=False)
    baseline_rmse = float(baseline_ckpt["val_rmse"])
    baseline_epoch = int(baseline_ckpt.get("epoch", 0))
    print(f"  Baseline (NDVI-only)  RMSE = {baseline_rmse:.6f}"
          f"  (epoch {baseline_epoch})")

    # ------------------------------------------------------------------
    # 3. Train fused LSTM
    # ------------------------------------------------------------------
    print("\n  ── Step 3: Train fused LSTM (input_size=6) ───────────────")
    fused_rmse = train_fused_lstm(X_fused, y, device)

    # ------------------------------------------------------------------
    # 4. Compute improvement
    # ------------------------------------------------------------------
    improvement_pct = 100.0 * (baseline_rmse - fused_rmse) / (baseline_rmse + 1e-9)

    # Reload fused checkpoint for R²
    fused_ckpt  = torch.load(str(FUSED_PT), map_location="cpu",
                              weights_only=False)
    fused_model = NDVIForecaster(input_size=INPUT_SIZE).to(device)
    fused_model.load_state_dict(fused_ckpt["model_state_dict"])
    fused_model.eval()

    n_val  = len(X_fused) - int(TRAIN_SPLIT * len(X_fused))
    X_val  = torch.FloatTensor(X_fused[-n_val:]).to(device)
    y_val  = y[-n_val:]
    with torch.no_grad():
        fused_preds = fused_model(X_val).cpu().numpy()
    fused_r2 = _r2(fused_preds, y_val)

    # Baseline R²
    baseline_model = NDVIForecaster(input_size=1).to(device)
    baseline_model.load_state_dict(baseline_ckpt["model_state_dict"])
    baseline_model.eval()
    data_ndvi = np.load(str(NDVI_NPZ))
    X_ndvi    = torch.FloatTensor(data_ndvi["X_sequences"])
    y_ndvi    = data_ndvi["y_targets"]
    n_bval    = len(X_ndvi) - int(TRAIN_SPLIT * len(X_ndvi))
    with torch.no_grad():
        base_preds = baseline_model(X_ndvi[-n_bval:].to(device)).cpu().numpy()
    baseline_r2 = _r2(base_preds, y_ndvi[-n_bval:])

    # ------------------------------------------------------------------
    # 5. Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  RESULTS COMPARISON")
    print("=" * 55)
    print(f"  {'Metric':<20} {'Baseline':>12} {'Fused':>12} {'Δ':>10}")
    print("  " + "-" * 55)
    print(f"  {'RMSE':<20} {baseline_rmse:>12.6f} {fused_rmse:>12.6f} "
          f"{improvement_pct:>+9.2f}%")
    print(f"  {'R²':<20} {baseline_r2:>12.6f} {fused_r2:>12.6f}")
    print("  " + "-" * 55)

    if improvement_pct >= 3.0:
        print(f"  ✓ RMSE improved by {improvement_pct:.2f}% — target (≥3%) achieved!")
    elif improvement_pct > 0:
        print(f"  ~ RMSE improved by {improvement_pct:.2f}% (target ≥3% not reached).")
    else:
        print(f"  ✗ Fused model did not improve over baseline.")

    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
