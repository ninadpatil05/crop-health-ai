"""
src/preprocessing/hyperspectral_loader.py
==========================================
Load, inspect, visualise and clean the Indian Pines hyperspectral cube.

Inputs
------
  data/indian_pines/Indian_pines.mat     (key: 'indian_pines'  → (145,145,200))
  data/indian_pines/Indian_pines_gt.mat  (key: 'indian_pines_gt' → (145,145))

Outputs
-------
  outputs/metrics/IP_falsecolor.png          – false-colour RGB composite
  outputs/metrics/IP_groundtruth.png         – ground-truth label map
  outputs/metrics/IP_spectral_signatures.png – mean spectral signature per class
  data/indian_pines/IP_cleaned.npy           – cube after water-band removal (145,145,176)
  data/indian_pines/band_indices.npy         – remaining band indices (176,)

Water absorption bands removed
-------------------------------
  [104,105,106,107,108] + range(150,164) + [220]  → 24 bands removed, 176 kept
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "indian_pines"
METRICS_DIR  = PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CUBE_PATH = DATA_DIR / "Indian_pines.mat"
GT_PATH   = DATA_DIR / "Indian_pines_gt.mat"

# Water absorption bands (0-indexed)
WATER_BANDS: list[int] = (
    [104, 105, 106, 107, 108]
    + list(range(150, 164))
    + [220]
)

# Human-readable class names for Indian Pines (classes 1-16, 0 = background)
CLASS_NAMES: dict[int, str] = {
    0:  "Background",
    1:  "Alfalfa",
    2:  "Corn-notill",
    3:  "Corn-mintill",
    4:  "Corn",
    5:  "Grass-pasture",
    6:  "Grass-trees",
    7:  "Grass-pasture-mowed",
    8:  "Hay-windrowed",
    9:  "Oats",
    10: "Soybean-notill",
    11: "Soybean-mintill",
    12: "Soybean-clean",
    13: "Wheat",
    14: "Woods",
    15: "Buildings-Grass-Trees-Drives",
    16: "Stone-Steel-Towers",
}


# ---------------------------------------------------------------------------
# Safe MAT loader
# ---------------------------------------------------------------------------
def load_mat(path: Path) -> np.ndarray:
    """
    Load a .mat file and return the first non-private key's array.

    scipy.io.loadmat stores the actual data under a variable-name key that
    differs between files.  Private keys start with '_' and are skipped.

    Parameters
    ----------
    path : Path to the .mat file.

    Returns
    -------
    numpy.ndarray  – the data array stored in the file.
    """
    mat = scipy.io.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith("_")][0]
    print(f"  [{path.name}]  key used: '{data_key}'")
    return mat[data_key]


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------
def plot_falsecolor(cube: np.ndarray, out_path: Path) -> None:
    """
    Create and save a false-colour RGB image using bands [29, 19, 9].

    The three chosen bands are normalised jointly to [0, 1] before display.

    Parameters
    ----------
    cube     : Hyperspectral cube of shape (H, W, C).
    out_path : Destination file path for the saved figure.
    """
    rgb = cube[:, :, [29, 19, 9]].astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title("Indian Pines — False-Colour RGB\n(Bands 29, 19, 9)", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_groundtruth(gt: np.ndarray, out_path: Path) -> None:
    """
    Plot the ground-truth label map with a discrete 'tab20' colormap.

    Class 0 (background) is rendered in black.  A colorbar listing class
    indices is added on the right.

    Parameters
    ----------
    gt       : Ground-truth array of shape (H, W) with integer class labels.
    out_path : Destination file path for the saved figure.
    """
    n_classes = int(gt.max()) + 1
    cmap = plt.get_cmap("tab20", n_classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(gt, cmap=cmap, vmin=0, vmax=n_classes - 1)
    ax.set_title("Indian Pines — Ground Truth Map", fontsize=12)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_ticks(range(n_classes))
    cbar.set_ticklabels([f"{i}: {CLASS_NAMES.get(i, str(i))}" for i in range(n_classes)],
                        fontsize=6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_spectral_signatures(cube: np.ndarray, gt: np.ndarray, out_path: Path) -> None:
    """
    Plot the mean spectral signature for every labelled class (class 0 skipped).

    Each line is the per-band mean of all pixels belonging to that class.

    Parameters
    ----------
    cube     : Hyperspectral cube (H, W, C).
    gt       : Ground-truth array (H, W).
    out_path : Destination file path for the saved figure.
    """
    classes = sorted(c for c in np.unique(gt) if c != 0)
    n_bands = cube.shape[2]
    band_idx = np.arange(n_bands)

    cmap = plt.get_cmap("tab20", len(classes))
    fig, ax = plt.subplots(figsize=(14, 5))

    for i, cls in enumerate(classes):
        mask  = gt == cls
        mean_sig = cube[mask].mean(axis=0).astype(np.float32)
        ax.plot(band_idx, mean_sig, color=cmap(i),
                linewidth=1.2, label=f"{cls}: {CLASS_NAMES.get(cls, str(cls))}")

    ax.set_title("Indian Pines — Mean Spectral Signatures per Class", fontsize=12)
    ax.set_xlabel("Band Index", fontsize=10)
    ax.set_ylabel("Mean Reflectance", fontsize=10)
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Full pipeline: load → inspect → visualise → clean → save.

    Returns
    -------
    cube        : Raw hyperspectral cube  (145, 145, 200)
    gt          : Ground-truth labels     (145, 145)
    cube_clean  : Water-band-removed cube (145, 145, 176)
    remaining   : List of kept band indices (length 176)
    """
    print("=" * 60)
    print("  Indian Pines Hyperspectral Loader")
    print("=" * 60)

    # ---- 1. Load ------------------------------------------------------------
    print("\n  Loading data ...")
    cube = load_mat(CUBE_PATH)
    gt   = load_mat(GT_PATH)

    print(f"\n  Cube shape  : {cube.shape}")
    print(f"  Cube dtype  : {cube.dtype}")
    print(f"  GT shape    : {gt.shape}")

    # ---- 2. Label statistics ------------------------------------------------
    unique_labels = np.unique(gt)
    print(f"\n  Unique GT labels  : {unique_labels.tolist()}")
    print(f"\n{'─'*52}")
    print(f"  {'Class':>5}  {'Name':<32}  {'Pixels':>7}")
    print(f"{'─'*52}")
    for lbl in unique_labels:
        px = int((gt == lbl).sum())
        name = CLASS_NAMES.get(int(lbl), "Unknown")
        print(f"  {lbl:>5}  {name:<32}  {px:>7,}")
    print(f"{'─'*52}")

    # ---- 3-5. EDA plots -----------------------------------------------------
    print("\n  Generating EDA figures ...")
    plot_falsecolor(cube, METRICS_DIR / "IP_falsecolor.png")
    plot_groundtruth(gt, METRICS_DIR / "IP_groundtruth.png")
    plot_spectral_signatures(cube, gt, METRICS_DIR / "IP_spectral_signatures.png")

    # ---- 6. Remove water absorption bands -----------------------------------
    remaining = [i for i in range(cube.shape[2]) if i not in WATER_BANDS]
    cube_clean = cube[:, :, remaining]

    print(f"\n  Band removal:")
    print(f"    Water bands removed : {len(WATER_BANDS)}")
    print(f"    Cube shape BEFORE   : {cube.shape}")
    print(f"    Cube shape AFTER    : {cube_clean.shape}")

    # ---- 7-8. Save artefacts ------------------------------------------------
    print("\n  Saving cleaned cube and band indices ...")
    np.save(DATA_DIR / "IP_cleaned.npy",    cube_clean)
    np.save(DATA_DIR / "band_indices.npy",  np.array(remaining, dtype=np.int32))
    print(f"    data/indian_pines/IP_cleaned.npy      {cube_clean.shape}")
    print(f"    data/indian_pines/band_indices.npy    ({len(remaining)},)")

    print("\n✓ Done.\n")
    return cube, gt, cube_clean, remaining


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_dataset()
