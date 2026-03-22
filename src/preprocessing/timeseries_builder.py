"""
src/preprocessing/timeseries_builder.py
=========================================
Build per-zone NDVI time series from dated Sentinel-2 stacks and package
them into sliding-window (X, y) sequences for LSTM / time-series training.

Behaviour
---------
* If **≥ 12** date-stamped stacks (``s2_stack_YYYYMMDD.npy``) are found in
  ``data/sentinel2/processed/``, real per-zone mean NDVI values are
  computed from each stack.
* If **< 12** stacks are found (or none exist), a 52-week synthetic NDVI
  series is generated for 100 zones so the rest of the pipeline can run
  uninterrupted.

Sliding-window approach
-----------------------
The model is trained to *forecast* by looking at a fixed-length history and
predicting the next two time steps.

  ┌──── window (8 steps) ────┐  ┌─ target (2 steps) ─┐
  t₀  t₁  t₂  t₃  t₄  t₅  t₆  t₇ │ t₈  t₉

The window slides by 1 step at a time across every zone's series:

  i=0 : X = series[0:8],  y = series[8:10]
  i=1 : X = series[1:9],  y = series[9:11]
  …

This means each zone of length T contributes  (T - 10)  training examples.
Using many overlapping windows maximises the number of training samples
from limited temporal data while teaching the model about short-range NDVI
dynamics.

Outputs
-------
  data/sentinel2/timeseries.npz          – X_sequences, y_targets, zone_ndvi
  outputs/metrics/ndvi_timeseries.png    – line plot for selected zones

Usage
-----
    python -m src.preprocessing.timeseries_builder
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROC_DIR      = PROJECT_ROOT / "data"    / "sentinel2" / "processed"
SAVE_PATH     = PROJECT_ROOT / "data"    / "sentinel2" / "timeseries.npz"
METRICS_DIR   = PROJECT_ROOT / "outputs" / "metrics"
PLOT_PATH     = METRICS_DIR / "ndvi_timeseries.png"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE   = 8     # input sequence length
TARGET_SIZE   = 2     # forecast horizon
STEP          = 1     # sliding-window stride
N_ZONES       = 100   # 10×10 spatial grid
GRID_ROWS     = 10
GRID_COLS     = 10
MIN_REAL_STACKS = 12  # threshold for synthetic fallback
PLOT_ZONES    = [0, 10, 25, 50, 75]


# ---------------------------------------------------------------------------
# 1. Discover dated stacks
# ---------------------------------------------------------------------------
def find_dated_stacks(proc_dir: Path) -> list[tuple[str, Path]]:
    """
    Scan *proc_dir* for files matching ``s2_stack_YYYYMMDD.npy``.

    Returns
    -------
    List of ``(date_str, path)`` tuples sorted chronologically.
    """
    pattern = re.compile(r"s2_stack_(\d{8})\.npy")
    results: list[tuple[str, Path]] = []
    for f in proc_dir.glob("*.npy"):
        m = pattern.match(f.name)
        if m:
            results.append((m.group(1), f))
    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# 2a. Real NDVI from stacks
# ---------------------------------------------------------------------------
def compute_zone_ndvi_real(
    dated_files: list[tuple[str, Path]],
    n_zones:     int = N_ZONES,
    grid_rows:   int = GRID_ROWS,
    grid_cols:   int = GRID_COLS,
) -> np.ndarray:
    """
    Compute mean NDVI for each spatial zone from real Sentinel-2 stacks.

    The image is divided into a ``grid_rows × grid_cols`` grid of zones.
    For each stack the NIR (band 3) and Red (band 2) channels are used:

        NDVI = (NIR - Red) / (NIR + Red + 1e-8)

    Parameters
    ----------
    dated_files : Sorted list of ``(date_str, path)`` from
                  :func:`find_dated_stacks`.
    n_zones     : Total number of zones (must equal grid_rows * grid_cols).
    grid_rows   : Number of rows in the spatial grid.
    grid_cols   : Number of columns in the spatial grid.

    Returns
    -------
    ndarray of shape ``(n_zones, num_dates)``, dtype float32.
    """
    num_dates = len(dated_files)
    zone_ndvi = np.zeros((n_zones, num_dates), dtype=np.float32)

    for t, (date_str, fpath) in enumerate(dated_files):
        s2     = np.load(str(fpath))          # (H, W, 6)
        H, W   = s2.shape[:2]
        nir    = s2[:, :, 3].astype(np.float32)
        red    = s2[:, :, 2].astype(np.float32)
        ndvi   = (nir - red) / (nir + red + 1e-8)

        zone_h = H // grid_rows
        zone_w = W // grid_cols

        for z in range(n_zones):
            r0 = (z // grid_cols) * zone_h
            c0 = (z  % grid_cols) * zone_w
            zone_ndvi[z, t] = float(
                ndvi[r0 : r0 + zone_h, c0 : c0 + zone_w].mean()
            )

        print(f"    [{t+1:>3}/{num_dates}] {date_str} — zone NDVI "
              f"range [{zone_ndvi[:, t].min():.3f}, {zone_ndvi[:, t].max():.3f}]")

    return zone_ndvi


# ---------------------------------------------------------------------------
# 2b. Synthetic NDVI (fallback)
# ---------------------------------------------------------------------------
def generate_synthetic_ndvi(
    n_zones: int = N_ZONES,
    weeks:   int = 52,
    seed:    int = 42,
) -> np.ndarray:
    """
    Generate synthetic per-zone NDVI time series for testing / demo purposes.

    Each zone starts from a fixed base NDVI value drawn uniformly from
    [0.3, 0.7] and evolves as a clipped random walk (σ = 0.01 per step).

    Parameters
    ----------
    n_zones : Number of spatial zones.
    weeks   : Length of the time series.
    seed    : NumPy random seed for reproducibility.

    Returns
    -------
    ndarray of shape ``(n_zones, weeks)``, dtype float32, values in [0.1, 0.9].
    """
    rng       = np.random.default_rng(seed)
    base_ndvi = rng.uniform(0.3, 0.7, n_zones)           # (n_zones,)
    zone_ndvi = np.zeros((n_zones, weeks), dtype=np.float32)

    for zone in range(n_zones):
        noise        = np.cumsum(rng.normal(0, 0.01, weeks))
        zone_series  = np.clip(base_ndvi[zone] + noise, 0.1, 0.9)
        zone_ndvi[zone] = zone_series.astype(np.float32)

    return zone_ndvi


# ---------------------------------------------------------------------------
# 3. Build sliding-window sequences
# ---------------------------------------------------------------------------
def build_sequences(
    zone_ndvi:   np.ndarray,
    window_size: int = WINDOW_SIZE,
    target_size: int = TARGET_SIZE,
    step:        int = STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping (X, y) pairs from the zone NDVI matrix.

    For each zone of length T, windows are created at positions:
        i = 0, step, 2*step, …  while (i + window_size + target_size) <= T

    Parameters
    ----------
    zone_ndvi   : ndarray of shape ``(n_zones, T)``.
    window_size : Number of time steps in the input sequence.
    target_size : Number of future steps to predict.
    step        : Stride of the sliding window.

    Returns
    -------
    X : float32 ndarray of shape ``(N, window_size, 1)``.
    y : float32 ndarray of shape ``(N, target_size)``.
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    n_zones, T = zone_ndvi.shape
    total_len  = window_size + target_size   # 10

    for zone in range(n_zones):
        series = zone_ndvi[zone]             # shape (T,)
        for i in range(0, len(series) - total_len + 1, step):
            X_list.append(series[i : i + window_size].reshape(window_size, 1))
            y_list.append(series[i + window_size : i + total_len])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
def plot_timeseries(
    zone_ndvi:   np.ndarray,
    plot_zones:  list[int] = PLOT_ZONES,
    save_path:   Path = PLOT_PATH,
    dated_files: list[tuple[str, Path]] | None = None,
) -> None:
    """
    Plot NDVI time series for a selection of zones and save as PNG.

    Parameters
    ----------
    zone_ndvi   : ndarray of shape ``(n_zones, T)``.
    plot_zones  : Zone indices to include in the plot.
    save_path   : Destination path for the PNG.
    dated_files : If provided, use date strings as x-tick labels.
    """
    T = zone_ndvi.shape[1]
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap    = plt.get_cmap("tab10")

    for i, z in enumerate(plot_zones):
        label = f"Zone {z}"
        ax.plot(x, zone_ndvi[z], color=cmap(i), linewidth=1.8,
                marker="o", markersize=3, label=label)

    if dated_files and len(dated_files) == T:
        dates = [d for d, _ in dated_files]
        step  = max(1, T // 12)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(dates[::step], rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xlabel("Week", fontsize=11)

    ax.set_ylabel("Mean NDVI", fontsize=11)
    ax.set_title("Per-Zone NDVI Time Series (selected zones)", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(0, 1)
    ax.axhline(0.6, color="green", linestyle="--", linewidth=1,
               alpha=0.6, label="Healthy threshold (0.6)")
    ax.axhline(0.2, color="red",   linestyle="--", linewidth=1,
               alpha=0.6, label="Stress threshold (0.2)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """End-to-end pipeline: discover → build zone NDVI → sequences → save → plot."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Sentinel-2 Time Series Builder")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Find dated stacks
    # ------------------------------------------------------------------
    dated_files = find_dated_stacks(PROC_DIR)
    print(f"\n  Found {len(dated_files)} dated stacks: "
          f"{[d for d, _ in dated_files]}")

    # ------------------------------------------------------------------
    # 2. Build zone NDVI matrix
    # ------------------------------------------------------------------
    if len(dated_files) >= MIN_REAL_STACKS:
        print(f"\n  Computing per-zone NDVI from {len(dated_files)} stacks …")
        zone_ndvi = compute_zone_ndvi_real(dated_files)
    else:
        print(f"\n  WARNING: Only {len(dated_files)} stacks found. "
              f"Generating synthetic time series.")
        zone_ndvi = generate_synthetic_ndvi(n_zones=N_ZONES, weeks=52)
        print(f"  Synthetic zone_ndvi shape : {zone_ndvi.shape}")

    print(f"\n  zone_ndvi shape : {zone_ndvi.shape}")
    print(f"  NDVI range      : [{zone_ndvi.min():.4f}, {zone_ndvi.max():.4f}]")
    print(f"  NDVI mean       : {zone_ndvi.mean():.4f}")

    # ------------------------------------------------------------------
    # 3. Build sliding-window sequences
    # ------------------------------------------------------------------
    print(f"\n  Building sequences (window={WINDOW_SIZE}, "
          f"target={TARGET_SIZE}, step={STEP}) …")
    X, y = build_sequences(zone_ndvi)
    print(f"  X shape         : {X.shape}   (samples, timesteps, features)")
    print(f"  y shape         : {y.shape}   (samples, forecast_horizon)")
    print(f"  X value range   : [{X.min():.4f}, {X.max():.4f}]")
    print(f"  y value range   : [{y.min():.4f}, {y.max():.4f}]")

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    np.savez(
        str(SAVE_PATH),
        X_sequences=X,
        y_targets=y,
        zone_ndvi=zone_ndvi,
    )
    print(f"\n  Saved → {SAVE_PATH}")

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    print(f"\n  Plotting NDVI time series for zones {PLOT_ZONES} …")
    plot_timeseries(
        zone_ndvi,
        plot_zones=PLOT_ZONES,
        save_path=PLOT_PATH,
        dated_files=dated_files if len(dated_files) >= MIN_REAL_STACKS else None,
    )

    # ------------------------------------------------------------------
    # 6. Sliding window explanation
    # ------------------------------------------------------------------
    print("""
  ── Sliding Window Explanation ──────────────────────────────────
  Each zone's NDVI series is divided into overlapping windows:

    Input  X : series[i   : i+8]   → shape (8, 1)
    Target y : series[i+8 : i+10]  → shape (2,)

  The window slides 1 step at a time. For a 52-week series this
  produces (52 - 10) = 42 examples per zone, and with 100 zones:
  42 × 100 = 4,200 training sequences.

  This dense overlap ensures the model sees every contiguous
  8-week pattern in the data, maximising sample efficiency and
  teaching it to forecast 2 weeks ahead from any history window.
  ────────────────────────────────────────────────────────────────
""")

    # ------------------------------------------------------------------
    # 7. Final summary
    # ------------------------------------------------------------------
    print("=" * 55)
    print("  Summary")
    print("=" * 55)
    print(f"  Zones            : {N_ZONES}")
    print(f"  Time steps       : {zone_ndvi.shape[1]}")
    print(f"  Total sequences  : {len(X)}")
    print(f"  X shape          : {X.shape}")
    print(f"  y shape          : {y.shape}")
    print(f"  Saved to         : {SAVE_PATH}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
