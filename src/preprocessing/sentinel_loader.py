"""
src/preprocessing/sentinel_loader.py
=====================================
Load Sentinel-2 imagery from either:
  • Copernicus Hub / ESA downloads  → .jp2 files
  • Google Earth Engine exports     → .tif / .TIF files

Band order in output stack:
  [0] Blue     B02 (10 m)
  [1] Green    B03 (10 m)
  [2] Red      B04 (10 m)
  [3] RedEdge  B05 (20 m, resampled to 10 m)
  [4] NIR      B08 (10 m)
  [5] SWIR     B11 (20 m, resampled to 10 m)

Output
------
  data/sentinel2/processed/s2_stack.npy          — latest stack (H, W, 6) float32
  data/sentinel2/processed/s2_stack_YYYYMMDD.npy — dated copy
  data/sentinel2/processed/meta.json             — CRS, transform, shape, band order
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform

# --- Project root (two parents up from this file) ------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR  = PROJECT_ROOT / "data" / "sentinel2" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "sentinel2" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# --- Band look-up table: name -> list of glob patterns (jp2 first, tif second) ---
BAND_PATTERNS: dict[str, list[str]] = {
    "Blue":    ["*_B02_10m.jp2", "*_B02.jp2", "*_B02.tif", "*_B2.tif"],
    "Green":   ["*_B03_10m.jp2", "*_B03.jp2", "*_B03.tif", "*_B3.tif"],
    "Red":     ["*_B04_10m.jp2", "*_B04.jp2", "*_B04.tif", "*_B4.tif"],
    "RedEdge": ["*_B05_20m.jp2", "*_B05.jp2", "*_B05.tif", "*_B5.tif"],
    "NIR":     ["*_B08_10m.jp2", "*_B08.jp2", "*_B08.tif", "*_B8.tif"],
    "SWIR":    ["*_B11_20m.jp2", "*_B11.jp2", "*_B11.tif", "*_B11.tif"],
}

BAND_ORDER = list(BAND_PATTERNS.keys())   # canonical ordering


# -----------------------------------------------------------------------------
def find_band(folder: Path, patterns: list[str]) -> Path | None:
    """
    Search *folder* (recursively) for the first file matching any of *patterns*.

    Patterns are tried in order so that higher-resolution Copernicus .jp2 files
    are preferred over coarser GEE .tif exports when both exist.

    Parameters
    ----------
    folder   : directory to search (searched recursively with rglob)
    patterns : list of glob patterns, e.g. ["*_B02_10m.jp2", "*_B02.tif"]

    Returns
    -------
    Path to the first matching file, or None if nothing is found.
    """
    for pattern in patterns:
        matches = sorted(folder.rglob(pattern))
        if matches:
            return matches[0]
    return None


# -----------------------------------------------------------------------------
def discover_bands(raw_dir: Path) -> dict[str, Path | None]:
    """
    Discover all six bands under *raw_dir*.

    Returns
    -------
    dict mapping band name → Path (or None if not found).
    Prints a status line for every band regardless of outcome.
    """
    found: dict[str, Path | None] = {}
    print("\n-- Band Discovery " + "-" * 44)
    for band_name, patterns in BAND_PATTERNS.items():
        path = find_band(raw_dir, patterns)
        found[band_name] = path
        status = f"  FOUND  -> {path.relative_to(raw_dir)}" if path else "  MISSING"
        print(f"  {band_name:<10} {status}")
    print("-" * 62)
    return found


# -----------------------------------------------------------------------------
def read_band(path: Path) -> tuple[np.ndarray, dict]:
    """
    Open a single raster band with rasterio and return (array, profile).

    The array is always 2-D (H, W) and dtype float32.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        profile["transform"] = src.transform
        profile["crs"]       = src.crs
        profile["height"]    = src.height
        profile["width"]     = src.width
    return arr, profile


# -----------------------------------------------------------------------------
def reproject_to_reference(
    src_path: Path,
    ref_profile: dict,
) -> np.ndarray:
    """
    Reproject / resample the raster at *src_path* to match *ref_profile*
    (CRS + transform + width + height).

    Uses bilinear resampling — appropriate for continuous reflectance values.

    Returns
    -------
    Reprojected array as float32 (H, W).
    """
    dst_arr = np.empty(
        (ref_profile["height"], ref_profile["width"]), dtype=np.float32
    )

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear,
        )

    return dst_arr


# -----------------------------------------------------------------------------
def build_stack(found: dict[str, Path | None], ref_profile: dict) -> np.ndarray:
    """
    Stack all found bands into a single (H, W, 6) float32 array.

    Bands that are not found are filled with zeros so the array remains
    fully shaped — downstream code can check meta.json for missing bands.

    Parameters
    ----------
    found      : band name -> Path mapping from :func:`discover_bands`
    ref_profile: rasterio profile of the reference band (Blue / B02)

    Returns
    -------
    np.ndarray of shape (H, W, 6), float32
    """
    H, W = ref_profile["height"], ref_profile["width"]
    stack = np.zeros((H, W, 6), dtype=np.float32)

    for idx, band_name in enumerate(BAND_ORDER):
        path = found[band_name]
        if path is None:
            print(f"  [WARN] {band_name}: missing — channel {idx} left as zeros")
            continue

        print(f"  Loading {band_name} from {path.name} ...", end=" ", flush=True)

        # Reference band needs no reprojection
        if band_name == "Blue":
            arr, _ = read_band(path)
        else:
            arr = reproject_to_reference(path, ref_profile)

        stack[:, :, idx] = arr
        print("done")

    return stack


# -----------------------------------------------------------------------------
def normalize(stack: np.ndarray) -> np.ndarray:
    """
    Normalize Sentinel-2 reflectance values.

    Copernicus Level-2A surface reflectance is stored as int16 with a
    scale factor of 10 000 (i.e. 10000 = 1.0 reflectance).
    GEE exports are already scaled but may have slightly different ranges.

    Steps
    -----
    1. Divide by 10 000.
    2. Clip to [0.0, 1.0].
    3. Replace any remaining NaN / negative values with 0.0.
    """
    out = stack / 10_000.0
    out = np.clip(out, 0.0, 1.0)
    out = np.where(np.isnan(out) | (out < 0.0), 0.0, out)
    return out.astype(np.float32)


# -----------------------------------------------------------------------------
def print_statistics(stack: np.ndarray) -> None:
    """Print per-band summary statistics to stdout."""
    H, W, _ = stack.shape
    total_pixels = H * W
    print("\n--- Stack Statistics ----------------------------------------")
    print(f"  Shape : {stack.shape}  (H={H}, W={W}, bands=6)")
    print(f"  dtype : {stack.dtype}")
    print(f"  {'Band':<12} {'Min':>8} {'Max':>8} {'Mean':>8} {'% Zeros':>9}")
    print(f"  {'-'*52}")
    for idx, band_name in enumerate(BAND_ORDER):
        ch = stack[:, :, idx]
        zero_pct = 100.0 * np.sum(ch == 0.0) / total_pixels
        print(
            f"  {band_name:<12} {ch.min():>8.4f} {ch.max():>8.4f}"
            f" {ch.mean():>8.4f} {zero_pct:>8.1f}%"
        )
    print("-" * 62)


# -----------------------------------------------------------------------------
def save_outputs(stack: np.ndarray, meta: dict) -> None:
    """
    Persist the processed stack and its metadata.

    Files written
    -------------
    • data/sentinel2/processed/s2_stack.npy
    • data/sentinel2/processed/s2_stack_YYYYMMDD.npy
    • data/sentinel2/processed/meta.json
    """
    today = datetime.now().strftime("%Y%m%d")

    npy_latest = PROC_DIR / "s2_stack.npy"
    npy_dated  = PROC_DIR / f"s2_stack_{today}.npy"

    np.save(npy_latest, stack)
    np.save(npy_dated,  stack)
    print(f"  Saved: {npy_latest.relative_to(PROJECT_ROOT)}")
    print(f"  Saved: {npy_dated.relative_to(PROJECT_ROOT)}")

    meta_path = PROC_DIR / "meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Saved: {meta_path.relative_to(PROJECT_ROOT)}")


# -----------------------------------------------------------------------------
def load_sentinel(raw_dir: Path = RAW_DIR) -> np.ndarray:
    """
    Main entry-point.  Discovers, loads, reprojects, stacks, normalises and
    saves the Sentinel-2 imagery found in *raw_dir*.

    Returns
    -------
    np.ndarray of shape (H, W, 6), dtype float32, values in [0, 1].
    Falls back to a reproducible synthetic 512x512 stack when fewer than
    6 bands can be found on disk.
    """
    print("=" * 62)
    print("  Sentinel-2 Loader")
    print(f"  Raw directory : {raw_dir}")
    print("=" * 62)

    found = discover_bands(raw_dir)
    n_found = sum(1 for v in found.values() if v is not None)

    # --- Fallback: synthetic data ----------------------------------------------
    if n_found < 6:
        missing = [k for k, v in found.items() if v is None]
        print(
            f"WARNING: Only {n_found}/6 bands found "
            f"(missing: {', '.join(missing)}).\n"
            "WARNING: Using synthetic data -- real bands not found"
        )
        rng = np.random.default_rng(42)
        stack = rng.random((512, 512, 6), dtype=np.float64).astype(np.float32)
        meta = {
            "synthetic": True,
            "shape": list(stack.shape),
            "band_order": BAND_ORDER,
            "crs": None,
            "transform": None,
        }
        print_statistics(stack)
        save_outputs(stack, meta)
        return stack

    # --- Real data path --------------------------------------------------------
    # Use Blue (B02) as the spatial reference
    ref_path = found["Blue"]
    _, ref_profile = read_band(ref_path)

    print("-- Loading & reprojecting bands " + "-" * 30)
    stack_raw = build_stack(found, ref_profile)

    print("\n--- Normalising ... ", end="", flush=True)
    stack = normalize(stack_raw)
    print("done")

    # Build metadata dict (JSON-serialisable)
    transform = ref_profile["transform"]
    meta = {
        "synthetic":   False,
        "shape":       list(stack.shape),
        "band_order":  BAND_ORDER,
        "crs":         str(ref_profile["crs"]),
        "transform": {
            "a": transform.a,
            "b": transform.b,
            "c": transform.c,
            "d": transform.d,
            "e": transform.e,
            "f": transform.f,
        },
        "files": {
            name: str(path.relative_to(raw_dir)) if path else None
            for name, path in found.items()
        },
    }

    print_statistics(stack)
    print("── Saving outputs ──────────────────────────────────────────")
    save_outputs(stack, meta)

    return stack


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stack = load_sentinel()
    print(f"\nReturned array shape : {stack.shape}")
    print(f"Returned array dtype : {stack.dtype}")
    print(f"Value range          : [{stack.min():.4f}, {stack.max():.4f}]")
