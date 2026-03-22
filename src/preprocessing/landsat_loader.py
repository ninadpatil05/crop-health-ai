"""
src/preprocessing/landsat_loader.py
=====================================
Load Landsat Collection 2 Level-2 Surface Reflectance bands, apply the
official scale factor, stack them into a single (H, W, 6) numpy array,
generate a false-colour preview, and save outputs.

Landsat 8/9 Collection 2 bands loaded
---------------------------------------
  Index 0 – Blue   (B2)
  Index 1 – Green  (B3)
  Index 2 – Red    (B4)
  Index 3 – NIR    (B5)
  Index 4 – SWIR1  (B6)
  Index 5 – SWIR2  (B7)

Scale factor (Collection 2)
----------------------------
  reflectance = raw * 0.0000275 - 0.2   clipped to [0, 1]
  Pixels where raw == 0  →  nodata, set to 0.0 after scaling.

Outputs
-------
  data/landsat/landsat_stack.npy        – float32 (H, W, 6)
  data/landsat/landsat_meta.json        – shape, crs, bands, date, scale
  data/landsat/preview_falsecolor.png   – NIR / Red / Green false-colour
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    warnings.warn("rasterio not found – will use synthetic data fallback.", ImportWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
LANDSAT_DIR   = PROJECT_ROOT / "data" / "landsat"
METRICS_DIR   = PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
LANDSAT_DIR.mkdir(parents=True, exist_ok=True)

BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
BAND_NUMS  = [2, 3, 4, 5, 6, 7]          # SR_B2 … SR_B7

# Collection 2 Level-2 scale / offset
SCALE_FACTOR = 0.0000275
SCALE_OFFSET = -0.2


# ---------------------------------------------------------------------------
# Band discovery
# ---------------------------------------------------------------------------
def find_landsat_band(folder: Path | str, band_num: int) -> Optional[Path]:
    """
    Search *folder* (recursively) for a Landsat band TIF, case-insensitively.

    Tries both long Collection-2 naming (``*_SR_B{n}.TIF``) and short
    variants.  Returns the first match or ``None``.

    Parameters
    ----------
    folder   : Directory to search.
    band_num : Band number (2–7 for SR products).
    """
    folder = Path(folder)
    patterns = [
        f"*_SR_B{band_num}.TIF",
        f"*_SR_B{band_num}.tif",
        f"*_B{band_num}.TIF",
        f"*_B{band_num}.tif",
    ]
    for p in patterns:
        matches = list(folder.rglob(p))
        if matches:
            return matches[0]
    return None


def extract_date(path: Path) -> str:
    """
    Pull the acquisition date from a Landsat filename.

    Landsat Collection-2 filenames follow the pattern::

        LC08_L2SP_148044_20230115_20230123_02_T1_SR_B4.TIF
                          ^^^^^^^^ ← acquisition date

    Returns ``"unknown"`` when the pattern is not found.
    """
    m = re.search(r"_(\d{8})_", path.name)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Loading & scaling
# ---------------------------------------------------------------------------
def load_and_scale_band(path: Path, ref_shape: Optional[tuple[int, int]] = None) -> np.ndarray:
    """
    Open a single Landsat TIF, apply Collection 2 scale factor, and
    optionally resample to *ref_shape*.

    Processing steps:

    1. Read band 1 as ``uint16`` (Landsat L2 native dtype).
    2. Record nodata pixels (raw value == 0).
    3. Apply ``reflectance = raw * 0.0000275 - 0.2``, clip to [0, 1].
    4. Re-apply nodata mask (set to 0.0).
    5. Resample to *ref_shape* via bilinear interpolation when provided.

    Parameters
    ----------
    path      : Path to the ``.TIF`` file.
    ref_shape : Target ``(height, width)`` for resampling; ``None`` = no resample.

    Returns
    -------
    numpy.ndarray  shape ``(H, W)``, dtype ``float32``.
    """
    with rasterio.open(path) as src:
        if ref_shape is None:
            raw = src.read(1)
        else:
            raw = src.read(
                1,
                out_shape=(ref_shape[0], ref_shape[1]),
                resampling=Resampling.bilinear,
            )

    nodata_mask = raw == 0
    reflectance = raw.astype(np.float32) * SCALE_FACTOR + SCALE_OFFSET
    reflectance = np.clip(reflectance, 0.0, 1.0)
    reflectance[nodata_mask] = 0.0
    return reflectance


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------
def save_falsecolor_preview(stack: np.ndarray, out_path: Path) -> None:
    """
    Create a false-colour composite (NIR / Red / Green) and save to disk.

    Pixel values are already in [0, 1]; a 2 % percentile stretch is applied
    per channel to improve visual contrast.

    Parameters
    ----------
    stack    : Float32 array ``(H, W, 6)`` in [Blue,Green,Red,NIR,SWIR1,SWIR2] order.
    out_path : Destination path for the PNG.
    """
    # NIR=idx3, Red=idx2, Green=idx1
    rgb = stack[:, :, [3, 2, 1]].copy()

    # Per-channel 2–98% percentile stretch for better contrast
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], (2, 98))
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title("Landsat — False-Colour Composite (NIR / Red / Green)", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Sentinel comparison (bonus)
# ---------------------------------------------------------------------------
def compare_with_sentinel(
    landsat_stack: np.ndarray,
    sentinel_stack: np.ndarray,
    out_dir: Path = METRICS_DIR,
) -> None:
    """
    Compute NDVI for Landsat and Sentinel stacks and plot them side by side.

    Assumes both stacks are in ``(H, W, C)`` format where:

    * Landsat  – NIR = index 3, Red = index 2
    * Sentinel – NIR = index 3, Red = index 2  (standard B2-B8 ordering)

    The figure is saved to ``out_dir/ndvi_comparison.png``.

    Parameters
    ----------
    landsat_stack  : Landsat float32 array ``(H, W, ≥4)``.
    sentinel_stack : Sentinel-2 float32 array ``(H, W, ≥4)``.
    out_dir        : Directory for the output figure.
    """
    def ndvi(stack: np.ndarray) -> np.ndarray:
        nir = stack[:, :, 3].astype(np.float32)
        red = stack[:, :, 2].astype(np.float32)
        denom = nir + red
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(denom > 0, (nir - red) / denom, 0.0)
        return result.astype(np.float32)

    ls_ndvi  = ndvi(landsat_stack)
    sen_ndvi = ndvi(sentinel_stack)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, arr, title in zip(
        axes,
        [ls_ndvi, sen_ndvi],
        ["Landsat NDVI", "Sentinel-2 NDVI"],
    ):
        im = ax.imshow(arr, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)

    plt.suptitle("NDVI Comparison — Landsat vs Sentinel-2", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "ndvi_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------
def make_synthetic_stack(h: int = 256, w: int = 256) -> np.ndarray:
    """
    Generate a plausible synthetic Landsat stack for testing.

    Values are drawn from a uniform distribution tailored per band to
    roughly match typical surface reflectance ranges.

    Parameters
    ----------
    h, w : Spatial dimensions.

    Returns
    -------
    numpy.ndarray  shape ``(h, w, 6)``, dtype ``float32``.
    """
    rng = np.random.default_rng(42)
    # Approximate reflectance ranges per band for a mixed scene
    ranges = [(0.02, 0.15), (0.02, 0.18), (0.02, 0.22),
              (0.10, 0.55), (0.05, 0.35), (0.02, 0.25)]
    bands = [rng.uniform(lo, hi, (h, w)).astype(np.float32) for lo, hi in ranges]
    return np.stack(bands, axis=-1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_dataset(data_dir: Path = LANDSAT_DIR) -> np.ndarray:
    """
    Full pipeline: discover → load → scale → stack → save → EDA.

    Parameters
    ----------
    data_dir : Folder containing the Landsat ``.TIF`` files.

    Returns
    -------
    numpy.ndarray  ``(H, W, 6)`` float32 Landsat stack.
    """
    print("=" * 60)
    print("  Landsat Collection 2 Loader")
    print(f"  Data directory : {data_dir}")
    print("=" * 60)

    # ---- 1. Discover bands ---------------------------------------------------
    print("\n  Discovering band files ...")
    band_paths: list[Optional[Path]] = []
    for bnum, bname in zip(BAND_NUMS, BAND_NAMES):
        p = find_landsat_band(data_dir, bnum)
        band_paths.append(p)
        status = p.name if p else "*** MISSING ***"
        print(f"    B{bnum} ({bname:>6}) : {status}")

    found = [p for p in band_paths if p is not None]

    # ---- 2. Load or fallback -----------------------------------------------
    if not found or not RASTERIO_OK:
        reason = "rasterio unavailable" if not RASTERIO_OK else "no band files found"
        print(f"\n  [WARN] {reason} — using synthetic 256×256×6 array.")
        stack = make_synthetic_stack()
        date_str = "synthetic"
        crs_str  = "N/A"
        ref_shape = (256, 256)
    else:
        # ---- 3. Load Reference (Band 2) to get shape & CRS ------------------
        print(f"\n  Loading and scaling bands (scale={SCALE_FACTOR}, offset={SCALE_OFFSET}) ...")
        with rasterio.open(found[0]) as src:
            ref_shape = (src.height, src.width)
            crs_str   = str(src.crs)

        date_str = extract_date(found[0])

        # ---- 4–5. Load all bands & resample to ref_shape --------------------
        band_arrays: list[np.ndarray] = []
        for bnum, bname, path in zip(BAND_NUMS, BAND_NAMES, band_paths):
            if path is None:
                print(f"    B{bnum} ({bname:>6}) : MISSING → filling with zeros")
                band_arrays.append(np.zeros(ref_shape, dtype=np.float32))
            else:
                arr = load_and_scale_band(path, ref_shape=ref_shape)
                band_arrays.append(arr)
                print(f"    B{bnum} ({bname:>6}) : min={arr.min():.4f}  max={arr.max():.4f}")

        # ---- 6. Stack -------------------------------------------------------
        stack = np.stack(band_arrays, axis=-1)   # (H, W, 6)

    # ---- 10. Per-band stats -------------------------------------------------
    print(f"\n  Stack shape : {stack.shape}  dtype={stack.dtype}")
    print(f"\n  {'Band':<6} {'Name':<8} {'Min':>8} {'Max':>8} {'Mean':>8}")
    print(f"  {'-'*44}")
    for i, name in enumerate(BAND_NAMES):
        b = stack[:, :, i]
        print(f"  {i:<6} {name:<8} {b.min():>8.4f} {b.max():>8.4f} {b.mean():>8.4f}")

    # ---- 7. Save stack ------------------------------------------------------
    npy_path = data_dir / "landsat_stack.npy"
    np.save(npy_path, stack)
    print(f"\n  Saved: {npy_path.relative_to(PROJECT_ROOT)}")

    # ---- 8. Save metadata ---------------------------------------------------
    meta = {
        "shape":        list(stack.shape),
        "crs":          crs_str,
        "bands":        {str(i): n for i, n in enumerate(BAND_NAMES)},
        "scale_factor": SCALE_FACTOR,
        "offset":       SCALE_OFFSET,
        "date_from_filename": date_str,
    }
    meta_path = data_dir / "landsat_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path.relative_to(PROJECT_ROOT)}")

    # ---- 9. False-colour preview --------------------------------------------
    print("\n  Generating false-colour preview ...")
    preview_path = data_dir / "preview_falsecolor.png"
    save_falsecolor_preview(stack, preview_path)

    print("\n✓ Done.\n")
    return stack


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_dataset()
