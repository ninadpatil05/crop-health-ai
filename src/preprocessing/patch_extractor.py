"""
src/preprocessing/patch_extractor.py
======================================
Extract 64×64 RGB patches from a Sentinel-2 image stack, label them using
NDVI thresholds, apply four-fold augmentation, and merge the resulting
manifest with the PlantVillage training CSV for combined model training.

Patch labelling
---------------
  mean_ndvi >= 0.6  →  label 0  (Healthy)
  0.2 <= mean_ndvi < 0.6  →  label 4  (Stress)
  mean_ndvi < 0.2          →  skipped  (bare soil / no vegetation)

Augmentation (4 versions per patch)
-------------------------------------
  original  │  horizontal flip  │  vertical flip  │  90° rotation

Outputs
-------
  data/sentinel2/patches/patch_NNNNN.npy  – individual patch arrays (float32)
  data/sentinel2/patches/patch_manifest.csv
  data/combined/train_combined.csv         – merged with PlantVillage train set

Usage
-----
    python -m src.preprocessing.patch_extractor
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

S2_STACK_PATH  = PROJECT_ROOT / "data" / "sentinel2" / "processed" / "s2_stack.npy"
NDVI_PATH      = PROJECT_ROOT / "outputs" / "maps" / "ndvi.npy"
PATCH_DIR      = PROJECT_ROOT / "data" / "sentinel2" / "patches"
MANIFEST_PATH  = PATCH_DIR / "patch_manifest.csv"
COMBINED_DIR   = PROJECT_ROOT / "data" / "combined"
COMBINED_CSV   = COMBINED_DIR / "train_combined.csv"
PV_TRAIN_CSV   = PROJECT_ROOT / "data" / "plantvillage" / "train.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATCH_SIZE  = 64
STRIDE      = 32
NDVI_MIN    = 0.2    # skip bare soil below this threshold
NDVI_HEALTH = 0.6    # above this → Healthy (label 0), else Stress (label 4)

LABEL_MAP: dict[int, str] = {0: "Healthy", 4: "Stress"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class PatchRecord(NamedTuple):
    """Holds a single augmented patch and its metadata."""
    array: np.ndarray       # shape (64, 64, 3), float32
    label: int
    label_name: str
    mean_ndvi: float
    source: str
    augmentation: str


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_patches(
    rgb: np.ndarray,
    ndvi: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """
    Slide a window over *rgb* and collect valid patches.

    Parameters
    ----------
    rgb        : ndarray of shape (H, W, 3) — float32 or uint16.
    ndvi       : ndarray of shape (H, W)    — float in [-1, 1].
    patch_size : Size of the square window in pixels.
    stride     : Step size of the sliding window.

    Returns
    -------
    (patches, labels, ndvis)
        patches : list of (patch_size, patch_size, 3) float32 arrays.
        labels  : list of int class indices.
        ndvis   : list of float mean NDVI values.
    """
    H, W = rgb.shape[:2]
    patches: list[np.ndarray] = []
    labels:  list[int]        = []
    ndvis:   list[float]      = []

    for r in range(0, H - patch_size, stride):
        for c in range(0, W - patch_size, stride):
            patch     = rgb[r : r + patch_size, c : c + patch_size, :]
            mean_ndvi = float(ndvi[r : r + patch_size, c : c + patch_size].mean())

            if mean_ndvi < NDVI_MIN:
                continue   # bare soil — skip

            label = 0 if mean_ndvi >= NDVI_HEALTH else 4

            patches.append(patch.astype(np.float32))
            labels.append(label)
            ndvis.append(mean_ndvi)

    return patches, labels, ndvis


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def augment_patches(
    patches: list[np.ndarray],
    labels:  list[int],
    ndvis:   list[float],
) -> list[PatchRecord]:
    """
    Apply four deterministic augmentations to each patch.

    Augmentations: original, horizontal flip, vertical flip, 90° rotation.

    Parameters
    ----------
    patches : Raw patches from :func:`extract_patches`.
    labels  : Corresponding class indices.
    ndvis   : Corresponding mean NDVI values.

    Returns
    -------
    List of :class:`PatchRecord` with 4× the input length.
    """
    augmented: list[PatchRecord] = []

    for patch, label, ndvi_val in zip(patches, labels, ndvis):
        label_name = LABEL_MAP[label]
        versions: list[tuple[np.ndarray, str]] = [
            (patch,             "original"),
            (np.fliplr(patch),  "hflip"),
            (np.flipud(patch),  "vflip"),
            (np.rot90(patch),   "rot90"),
        ]
        for arr, aug_name in versions:
            augmented.append(
                PatchRecord(
                    array=arr,
                    label=label,
                    label_name=label_name,
                    mean_ndvi=ndvi_val,
                    source="sentinel2",
                    augmentation=aug_name,
                )
            )

    return augmented


# ---------------------------------------------------------------------------
# Save patches + manifest
# ---------------------------------------------------------------------------
def save_patches(
    augmented: list[PatchRecord],
    patch_dir: Path = PATCH_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> pd.DataFrame:
    """
    Persist each augmented patch as a .npy file and write the manifest CSV.

    Parameters
    ----------
    augmented     : Output of :func:`augment_patches`.
    patch_dir     : Directory to store .npy files and the manifest.
    manifest_path : Destination path for the CSV manifest.

    Returns
    -------
    DataFrame with columns: image_path, label, label_name, mean_ndvi, source.
    """
    patch_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for idx, rec in enumerate(augmented):
        fname = f"patch_{idx:05d}.npy"
        fpath = patch_dir / fname
        np.save(str(fpath), rec.array)

        records.append(
            {
                "image_path": str(fpath),
                "label":      rec.label,
                "label_name": rec.label_name,
                "mean_ndvi":  round(rec.mean_ndvi, 6),
                "source":     rec.source,
                "augmentation": rec.augmentation,
            }
        )

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_df


# ---------------------------------------------------------------------------
# Merge with PlantVillage
# ---------------------------------------------------------------------------
def merge_with_plantvillage(
    manifest_df: pd.DataFrame,
    pv_train_csv: Path = PV_TRAIN_CSV,
    combined_csv: Path = COMBINED_CSV,
) -> pd.DataFrame:
    """
    Concatenate the Sentinel-2 patch manifest with the PlantVillage training CSV.

    Only the columns ``image_path``, ``label``, and ``source`` are kept so
    the schema is uniform across both datasets.

    Parameters
    ----------
    manifest_df  : Sentinel-2 patch manifest from :func:`save_patches`.
    pv_train_csv : Path to PlantVillage train.csv.
    combined_csv : Destination path for the merged CSV.

    Returns
    -------
    Combined DataFrame.
    """
    pv_train = pd.read_csv(pv_train_csv)
    pv_train["source"] = "plantvillage"

    patches_df = manifest_df[["image_path", "label", "source"]].copy()
    pv_subset  = pv_train[["image_path", "label", "source"]].copy()

    combined = pd.concat([pv_subset, patches_df], ignore_index=True)

    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_csv, index=False)
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """
    End-to-end pipeline: load → extract → augment → save → merge.
    """
    print("=" * 55)
    print("  Sentinel-2 Patch Extractor")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print(f"\n  Loading Sentinel-2 stack  : {S2_STACK_PATH}")
    s2   = np.load(str(S2_STACK_PATH))          # (H, W, 6)
    print(f"  s2 shape                  : {s2.shape}")

    print(f"  Loading NDVI map          : {NDVI_PATH}")
    ndvi = np.load(str(NDVI_PATH))              # (H, W)
    print(f"  NDVI shape                : {ndvi.shape}")

    # ------------------------------------------------------------------
    # 2. Extract RGB patches (bands 2,1,0 → R,G,B)
    # ------------------------------------------------------------------
    rgb = s2[:, :, [2, 1, 0]]                   # (H, W, 3)
    print(f"\n  Extracting {PATCH_SIZE}×{PATCH_SIZE} patches "
          f"(stride={STRIDE}) …")

    patches, labels, ndvis = extract_patches(rgb, ndvi)

    print(f"  Raw patches extracted     : {len(patches)}")

    # Label distribution
    for lbl, name in sorted(LABEL_MAP.items()):
        count = labels.count(lbl)
        pct   = 100 * count / max(len(labels), 1)
        print(f"    label {lbl}  ({name:8s})   : {count:6d}  ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # 3. Augmentation (×4)
    # ------------------------------------------------------------------
    print(f"\n  Augmenting patches (×4) …")
    augmented = augment_patches(patches, labels, ndvis)
    print(f"  Total augmented patches   : {len(augmented)}")

    # ------------------------------------------------------------------
    # 4. Save .npy files + manifest
    # ------------------------------------------------------------------
    print(f"\n  Saving patches to {PATCH_DIR} …")
    manifest_df = save_patches(augmented)
    print(f"  Manifest saved            : {MANIFEST_PATH}")

    # ------------------------------------------------------------------
    # 5. Merge with PlantVillage
    # ------------------------------------------------------------------
    print(f"\n  Merging with PlantVillage train set …")
    combined = merge_with_plantvillage(manifest_df)
    print(f"  Combined CSV saved        : {COMBINED_CSV}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    print(f"  Raw patches extracted     : {len(patches)}")
    print(f"  Augmented patches         : {len(augmented)}")
    for lbl, name in sorted(LABEL_MAP.items()):
        count = sum(1 for r in augmented if r.label == lbl)
        print(f"    {name:14s} (label {lbl})  : {count:6d}")
    print(f"  Combined dataset rows     : {len(combined)}")
    pv_rows = (combined["source"] == "plantvillage").sum()
    s2_rows = (combined["source"] == "sentinel2").sum()
    print(f"    PlantVillage            : {pv_rows}")
    print(f"    Sentinel-2 patches      : {s2_rows}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
