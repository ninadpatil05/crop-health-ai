"""
src/indices/compute_indices.py
==============================
Compute six remote-sensing spectral indices from a pre-processed
Sentinel-2 multispectral stack (H, W, 6) float32 in [0, 1].

Band layout expected in the input stack
---------------------------------------
  [0] Blue     B02
  [1] Green    B03
  [2] Red      B04
  [3] RedEdge  B05
  [4] NIR      B08
  [5] SWIR     B11

Indices computed
----------------
  NDVI  — Normalised Difference Vegetation Index
  NDRE  — Normalised Difference Red-Edge
  SAVI  — Soil-Adjusted Vegetation Index
  EVI   — Enhanced Vegetation Index
  NDWI  — Normalised Difference Water Index
  BSI   — Bare Soil Index

Outputs
-------
  outputs/maps/ndvi.npy  … bsi.npy           — per-pixel index maps (H, W)
  outputs/metrics/vegetation_indices.png      — 2×3 diagnostic figure
"""

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot import — non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

STACK_PATH  = PROJECT_ROOT / "data" / "sentinel2" / "processed" / "s2_stack.npy"
MAPS_DIR    = PROJECT_ROOT / "outputs" / "maps"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

MAPS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe(arr: np.ndarray) -> np.ndarray:
    """Replace non-finite values and clip to [-1, 1]."""
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


def _print_stats(name: str, arr: np.ndarray) -> None:
    """Print one-line summary statistics for an index map."""
    print(
        f"  {name:<6} | mean={arr.mean():+.4f} | std={arr.std():.4f} "
        f"| min={arr.min():+.4f} | max={arr.max():+.4f}"
    )


# ---------------------------------------------------------------------------
# Index computations
# ---------------------------------------------------------------------------
def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Range [-1, 1].  High positive values → dense green vegetation.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (nir - red) / (nir + red)
    return _safe(result)


def compute_ndre(nir: np.ndarray, re: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Red-Edge.

    NDRE = (NIR - RedEdge) / (NIR + RedEdge)

    More sensitive to chlorophyll content than NDVI; useful for dense crops.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (nir - re) / (nir + re)
    return _safe(result)


def compute_savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """
    Soil-Adjusted Vegetation Index.

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)   where L = 0.5

    Reduces the influence of bare soil background on vegetation signal.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = ((nir - red) / (nir + red + L)) * (1.0 + L)
    return _safe(result)


def compute_evi(
    nir: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
) -> np.ndarray:
    """
    Enhanced Vegetation Index.

    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

    Corrects for atmospheric and canopy background effects; less saturated
    than NDVI in high-biomass areas.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = G * (nir - red) / (nir + C1 * red - C2 * blue + L)
    return _safe(result)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Water Index (McFeeters 1996).

    NDWI = (Green - NIR) / (Green + NIR)

    Positive values indicate open water bodies or flooded areas.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (green - nir) / (green + nir)
    return _safe(result)


def compute_bsi(
    red: np.ndarray,
    swir: np.ndarray,
    nir: np.ndarray,
    blue: np.ndarray,
) -> np.ndarray:
    """
    Bare Soil Index.

    BSI = ((Red + SWIR) - (NIR + Blue)) / ((Red + SWIR) + (NIR + Blue))

    High positive values indicate exposed, bare, or degraded soil with
    little or no vegetation cover.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator   = (red + swir) - (nir + blue)
        denominator = (red + swir) + (nir + blue)
        result = numerator / denominator
    return _safe(result)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
_INDEX_VIS: list[dict[str, Any]] = [
    {"key": "NDVI", "cmap": "RdYlGn",  "vmin": -1, "vmax": 1},
    {"key": "NDRE", "cmap": "RdYlGn",  "vmin": -1, "vmax": 1},
    {"key": "SAVI", "cmap": "RdYlGn",  "vmin": -1, "vmax": 1},
    {"key": "EVI",  "cmap": "RdYlGn",  "vmin": -1, "vmax": 1},
    {"key": "NDWI", "cmap": "Blues_r", "vmin": -1, "vmax": 1},
    {"key": "BSI",  "cmap": "YlOrBr",  "vmin": -1, "vmax": 1},
]


def plot_indices(indices: dict[str, np.ndarray], out_path: Path) -> None:
    """
    Save a 2×3 diagnostic figure showing all six index maps.

    Parameters
    ----------
    indices  : dict mapping index name → (H, W) float32 array
    out_path : destination PNG file
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Sentinel-2 Spectral Indices", fontsize=15, fontweight="bold", y=1.01)

    for ax, cfg in zip(axes.flat, _INDEX_VIS):
        arr = indices[cfg["key"]]
        im  = ax.imshow(arr, cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
        ax.set_title(cfg["key"], fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------
def compute_all(stack_path: Path = STACK_PATH) -> dict[str, np.ndarray]:
    """
    Load the Sentinel-2 stack, compute all six indices, save .npy files
    and a diagnostic PNG, and return a dict of index arrays.

    Parameters
    ----------
    stack_path : path to the s2_stack.npy file produced by sentinel_loader.py

    Returns
    -------
    dict mapping index name → (H, W) float32 ndarray, values in [-1, 1].
    """
    print("=" * 62)
    print("  Spectral Index Calculator")
    print(f"  Stack : {stack_path}")
    print("=" * 62)

    if not stack_path.exists():
        raise FileNotFoundError(
            f"Stack not found at {stack_path}. "
            "Run src/preprocessing/sentinel_loader.py first."
        )

    print("\n  Loading stack ...", end=" ", flush=True)
    stack = np.load(stack_path)           # (H, W, 6) float32
    print(f"done  shape={stack.shape}")

    # Unpack spectral bands
    blue  = stack[:, :, 0]
    green = stack[:, :, 1]
    red   = stack[:, :, 2]
    re    = stack[:, :, 3]   # RedEdge
    nir   = stack[:, :, 4]
    swir  = stack[:, :, 5]

    print("\n  Computing indices ...")
    indices: dict[str, np.ndarray] = {
        "NDVI": compute_ndvi(nir, red),
        "NDRE": compute_ndre(nir, re),
        "SAVI": compute_savi(nir, red),
        "EVI":  compute_evi(nir, red, blue),
        "NDWI": compute_ndwi(green, nir),
        "BSI":  compute_bsi(red, swir, nir, blue),
    }

    # ---- statistics --------------------------------------------------------
    print("\n--- Index Statistics -----------------------------------------")
    print(f"  {'Name':<6} | {'Mean':>9} | {'Std':>7} | {'Min':>7} | {'Max':>7}")
    print("  " + "-" * 56)
    for name, arr in indices.items():
        _print_stats(name, arr)
    print("-" * 62)

    # ---- save .npy ---------------------------------------------------------
    print("\n  Saving index maps ...")
    for name, arr in indices.items():
        out = MAPS_DIR / f"{name.lower()}.npy"
        np.save(out, arr)
        print(f"    {out.relative_to(PROJECT_ROOT)}")

    # ---- figure ------------------------------------------------------------
    print("\n  Generating figure ...")
    fig_path = METRICS_DIR / "vegetation_indices.png"
    plot_indices(indices, fig_path)

    print("\n✓ Done.\n")
    return indices


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    compute_all()
