"""
src/preprocessing/plantvillage_loader.py
=========================================
Scan the PlantVillage dataset, map every folder to one of 5 disease
categories, split into train / val / test sets, save CSVs, and generate
exploratory-data-analysis (EDA) figures.

Folder naming conventions handled
----------------------------------
  Flat   : data/plantvillage/Tomato___Early_blight/img.jpg
  Nested : data/plantvillage/color/Tomato___Early_blight/img.jpg

Category mapping (substring match, case-insensitive)
------------------------------------------------------
  0  Healthy        → 'healthy'
  1  Fungal Disease → 'blight', 'mold', 'rust', 'scab', 'rot'
  2  Bacterial      → 'bacterial', 'angular', 'canker', 'citrus_greening'
  3  Pest Damage    → 'mite', 'miner', 'aphid', 'whitefly', 'leafhopper'
  4  Stress         → everything else

Outputs
-------
  data/plantvillage/train.csv
  data/plantvillage/val.csv
  data/plantvillage/test.csv
  outputs/metrics/sample_grid.png
  outputs/metrics/class_distribution.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "plantvillage"
METRICS_DIR  = PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------
CATEGORY_MAP: dict[int, tuple[str, list[str]]] = {
    0: ("Healthy",        ["healthy"]),
    1: ("Fungal Disease", ["blight", "mold", "rust", "scab", "rot"]),
    2: ("Bacterial",      ["bacterial", "angular", "canker", "citrus_greening"]),
    3: ("Pest Damage",    ["mite", "miner", "aphid", "whitefly", "leafhopper"]),
    4: ("Stress",         []),          # catch-all
}

LABEL_NAMES: dict[int, str] = {k: v[0] for k, v in CATEGORY_MAP.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def classify_folder(folder_name: str) -> tuple[str, int]:
    """
    Map a PlantVillage folder name to a (category_name, label) pair.

    Matching is done left-to-right, first match wins.  'Stress' (label 4)
    is the catch-all when nothing else matches.

    Parameters
    ----------
    folder_name : raw directory name, e.g. 'Tomato___Early_blight'

    Returns
    -------
    (category_name, label)
    """
    lower = folder_name.lower()
    for label in range(5):           # iterate in priority order 0→4
        name, keywords = CATEGORY_MAP[label]
        if label == 4:               # Stress is always the catch-all
            return name, label
        if any(kw in lower for kw in keywords):
            return name, label
    return LABEL_NAMES[4], 4        # should never reach here


def discover_images(root: Path) -> pd.DataFrame:
    """
    Recursively find all .jpg / .JPG / .png images under *root*.

    Handles both flat and colour-subfolder layouts.

    Returns
    -------
    DataFrame with columns: image_path, original_class, category, label
    """
    records: list[dict] = []
    image_extensions = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")

    # Collect all image files
    all_images: list[Path] = []
    for pattern in image_extensions:
        all_images.extend(root.rglob(pattern))

    if not all_images:
        print(f"\n[ERROR] No images found under: {root}")
        print("Folder contents:")
        for item in sorted(root.iterdir()):
            print(f"  {'DIR' if item.is_dir() else 'FILE'}: {item.name}")
        raise ValueError(
            f"No images found in {root}. "
            "Please verify the PlantVillage dataset is placed correctly."
        )

    for img_path in all_images:
        # The 'original class' folder is the immediate parent of the image
        # (or parent of parent if there is a colour-mode subfolder)
        parent = img_path.parent
        # Skip if parent is the root itself (shouldn't happen but be safe)
        if parent == root:
            folder_name = "unknown"
        else:
            # Walk up until we find a folder whose name contains '___'
            # (PlantVillage convention), or default to immediate parent
            folder_name = parent.name
            for ancestor in img_path.parents:
                if ancestor == root:
                    break
                if "___" in ancestor.name or ancestor == parent:
                    folder_name = ancestor.name
                    break

        category, label = classify_folder(folder_name)
        records.append(
            {
                "image_path":     str(img_path.resolve()),
                "original_class": folder_name,
                "category":       category,
                "label":          label,
            }
        )

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    random_state: int  = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70 / 15 / 15 train / val / test split.

    Classes with fewer than 2 samples are dropped before splitting to avoid
    stratification errors.

    Parameters
    ----------
    df           : full DataFrame from :func:`discover_images`
    train_ratio  : fraction for training (default 0.70)
    val_ratio    : fraction for validation (default 0.15)
    random_state : reproducibility seed

    Returns
    -------
    (train_df, val_df, test_df)
    """
    # Drop rare classes
    counts = df["label"].value_counts()
    rare   = counts[counts < 2].index.tolist()
    if rare:
        rare_names = [LABEL_NAMES.get(r, str(r)) for r in rare]
        warnings.warn(
            f"Dropping classes with <2 samples: {rare_names}", UserWarning
        )
        print(f"\n[WARN] Dropping rare class(es): {rare_names}")
        df = df[~df["label"].isin(rare)].copy()

    # First cut: train vs temp (val+test)
    test_val_ratio = 1.0 - train_ratio                  # 0.30
    train_df, temp_df = train_test_split(
        df,
        test_size=test_val_ratio,
        stratify=df["label"],
        random_state=random_state,
    )

    # Second cut: val vs test (50/50 of the temp 30%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """
    Bar chart of image counts per category across the full dataset.
    Saved to *out_path*.
    """
    counts = (
        df.groupby(["label", "category"])
        .size()
        .reset_index(name="count")
        .sort_values("label")
    )

    colors = ["#4CAF50", "#FF7043", "#42A5F5", "#AB47BC", "#FFA726"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        counts["category"],
        counts["count"],
        color=[colors[int(l)] for l in counts["label"]],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar_label(bars, padding=4, fontsize=9, fmt="%d")
    ax.set_title("PlantVillage — Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, counts["count"].max() * 1.12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_sample_grid(df: pd.DataFrame, out_path: Path, n_per_class: int = 5) -> None:
    """
    Draw a 5×5 grid showing *n_per_class* sample images for each category.

    Rows = categories (label 0–4), columns = sample images.
    Saved to *out_path*.
    """
    labels = sorted(df["label"].unique())
    n_rows = len(labels)
    n_cols = n_per_class

    fig = plt.figure(figsize=(n_cols * 2.8, n_rows * 2.8))
    fig.suptitle(
        "PlantVillage — Sample Images per Category",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for row_idx, lbl in enumerate(labels):
        subset = df[df["label"] == lbl].sample(
            min(n_per_class, len(df[df["label"] == lbl])),
            random_state=42,
        )
        category_name = LABEL_NAMES[lbl]

        for col_idx, (_, record) in enumerate(subset.iterrows()):
            ax_idx = row_idx * n_cols + col_idx + 1
            ax = fig.add_subplot(n_rows, n_cols, ax_idx)
            try:
                img = Image.open(record["image_path"]).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("#cccccc")
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(f"[{lbl}] {category_name}", fontsize=9,
                             fontweight="bold", loc="left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_split_distribution(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """Print a per-category count table for each split."""
    print("\n--- Class Distribution per Split " + "-" * 30)
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n  {split_name} ({len(split_df)} images):")
        print(f"    {'Label':<5} {'Category':<18} {'Count':>7} {'%':>7}")
        print("    " + "-" * 40)
        for lbl in sorted(split_df["label"].unique()):
            cat = LABEL_NAMES[lbl]
            cnt = (split_df["label"] == lbl).sum()
            pct = 100.0 * cnt / len(split_df)
            print(f"    {lbl:<5} {cat:<18} {cnt:>7} {pct:>6.1f}%")
    print("-" * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: discover → classify → split → save CSVs → EDA figures.

    Parameters
    ----------
    data_dir : root of the PlantVillage dataset

    Returns
    -------
    (train_df, val_df, test_df)
    """
    print("=" * 62)
    print("  PlantVillage Loader")
    print(f"  Data directory : {data_dir}")
    print("=" * 62)

    # ---- Discover images ---------------------------------------------------
    print("\n  Scanning for images ...")
    df = discover_images(data_dir)
    print(f"  Total images found : {len(df):,}")

    # ---- Summary per category ----------------------------------------------
    print("\n--- Images per Category " + "-" * 38)
    print(f"  {'Label':<5} {'Category':<18} {'Count':>8} {'Folders':>8}")
    print("  " + "-" * 42)
    for lbl in sorted(df["label"].unique()):
        subset  = df[df["label"] == lbl]
        folders = subset["original_class"].nunique()
        print(f"  {lbl:<5} {LABEL_NAMES[lbl]:<18} {len(subset):>8,} {folders:>8}")
    print("-" * 62)

    # ---- Split -------------------------------------------------------------
    print("\n  Splitting 70 / 15 / 15 ...")
    train_df, val_df, test_df = split_dataset(df)
    print_split_distribution(train_df, val_df, test_df)

    # ---- Save CSVs ---------------------------------------------------------
    print("\n  Saving CSVs ...")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = data_dir / f"{name}.csv"
        split_df.to_csv(out, index=False)
        print(f"    {out.relative_to(PROJECT_ROOT)}  ({len(split_df)} rows)")

    # ---- EDA figures -------------------------------------------------------
    print("\n  Generating EDA figures ...")
    plot_class_distribution(df, METRICS_DIR / "class_distribution.png")
    plot_sample_grid(df, METRICS_DIR / "sample_grid.png")

    print("\n✓ Done.\n")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_df, val_df, test_df = build_dataset()
