"""
src/training/cnn_model.py
==========================
5-class crop disease / stress classifier built on a pretrained ResNet-18
backbone (ImageNet weights, partial fine-tuning).

Why ResNet-18 over training from scratch?
-----------------------------------------
PlantVillage has ~108 k images spread across 5 broad categories — large
enough to avoid trivial overfitting but not large enough to reliably train
a deep CNN from random initialisation in reasonable time on a single GPU.
ResNet-18 pre-trained on ImageNet already "knows" low-level visual features
(edges, textures, colour gradients) that transfer strongly to plant imagery.
By freezing the early layers and fine-tuning only layer3, layer4, and the
newly added head we get:

  1. **Faster convergence** – only ~3 M of the 11 M parameters are updated.
  2. **Better generalisation** – frozen early layers act as a fixed, powerful
     feature extractor, reducing the effective model capacity and therefore
     the risk of overfitting.
  3. **Lower data hunger** – transfer learning routinely matches or beats
     scratch training with 5–10× less labelled data.

Architecture summary
---------------------
  ResNet-18 (frozen backbone)
    └─ layer3  (unfrozen — mid-level texture features)
    └─ layer4  (unfrozen — high-level semantic features)
    └─ fc  →  Linear(512,256) → ReLU → Dropout(0.4) → Linear(256,5)

Classes
-------
  0 Healthy | 1 Fungal Disease | 2 Bacterial | 3 Pest Damage | 4 Stress
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import torchvision.models as models
import torchvision.transforms as T

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES  = 5
CLASS_NAMES  = ["Healthy", "Fungal Disease", "Bacterial", "Pest Damage", "Stress"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(split: str = "train") -> T.Compose:
    """
    Return the torchvision transform pipeline for a given dataset split.

    Training uses aggressive augmentation (flips, rotation, colour jitter)
    to improve generalisation.  Validation uses a deterministic centre-crop.

    Parameters
    ----------
    split : ``'train'`` or ``'val'`` / ``'test'``.

    Returns
    -------
    torchvision.transforms.Compose
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return T.Compose([
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
            T.ToTensor(),
            normalize,
        ])
    else:  # val / test
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CropDataset(Dataset):
    """
    PyTorch Dataset that reads image paths and labels from a CSV file.

    Supports three file types transparently:

    * ``.jpg`` / ``.jpeg`` / ``.png``  – opened with ``PIL.Image``.
    * ``.npy``                          – loaded with ``numpy``; expected shape
      ``(64, 64, 3)`` float32 in [0, 1]; converted to a PIL image before
      applying transforms.

    Corrupted or unreadable files are skipped silently (a ``None`` sentinel
    is returned and filtered out by the collate function).

    Parameters
    ----------
    csv_path  : Path to a CSV with at least ``image_path`` and ``label`` columns.
    transform : torchvision transform to apply after loading.
    """

    def __init__(self, csv_path: str | Path, transform: Optional[T.Compose] = None) -> None:
        self.df        = pd.read_csv(csv_path)
        self.transform = transform
        self._skipped  = 0

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def _load_image(self, path: str) -> Optional[Image.Image]:
        """
        Load a single image from *path*, returning ``None`` on failure.

        Handles both raster image files and ``.npy`` float arrays.
        """
        p = Path(path)
        try:
            if p.suffix.lower() == ".npy":
                arr = np.load(str(p))          # (H, W, 3) float32 [0,1]
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                return Image.fromarray(arr, mode="RGB")
            else:
                return Image.open(str(p)).convert("RGB")
        except Exception:
            return None

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Optional[tuple[torch.Tensor, int]]:
        row   = self.df.iloc[idx]
        label = int(row["label"])
        img   = self._load_image(str(row["image_path"]))

        if img is None:
            self._skipped += 1
            # Return a black image so the batch stays intact
            img = Image.new("RGB", (224, 224), color=0)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Build a partially fine-tuned ResNet-18 for crop disease classification.

    Strategy
    --------
    1. Load ResNet-18 with ImageNet-1K weights.
    2. Freeze **all** parameters (no gradient flow through frozen layers).
    3. Selectively unfreeze ``layer3`` and ``layer4`` so the network can
       adapt high-level features to plant imagery.
    4. Replace the final fully-connected layer with a custom head that
       outputs ``num_classes`` logits via a bottleneck with dropout.

    Parameters
    ----------
    num_classes : Number of output classes (default 5).
    pretrained  : Load ImageNet weights when ``True``.

    Returns
    -------
    torch.nn.Module  – model ready for ``.train()`` / ``.eval()``.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    backbone = models.resnet18(weights=weights)

    # Step 1 – freeze everything
    for p in backbone.parameters():
        p.requires_grad = False

    # Step 2 – unfreeze layer3 and layer4
    for p in backbone.layer3.parameters():
        p.requires_grad = True
    for p in backbone.layer4.parameters():
        p.requires_grad = True

    # Step 3 – replace classification head
    in_features = backbone.fc.in_features   # 512 for ResNet-18
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )

    return backbone


# Public alias — allows: from src.training.cnn_model import CropDiseaseCNN
CropDiseaseCNN = build_model


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------
def get_dataloaders(
    train_csv: str | Path,
    val_csv:   str | Path,
    test_csv:  Optional[str | Path] = None,
    batch_size: int = 32,
) -> dict[str, DataLoader]:
    """
    Build train / val (/ test) DataLoaders from CSV files.

    ``num_workers`` is set to 0 on Windows to avoid multiprocessing
    spawn issues; 4 on POSIX systems.

    Parameters
    ----------
    train_csv  : Path to training CSV.
    val_csv    : Path to validation CSV.
    test_csv   : Optional path to test CSV.
    batch_size : Mini-batch size (default 32).

    Returns
    -------
    dict with keys ``'train'``, ``'val'``, and optionally ``'test'``,
    each mapping to a ``DataLoader``.
    """
    nw = 0 if platform.system() == "Windows" else 4

    train_ds = CropDataset(train_csv, transform=get_transforms("train"))
    val_ds   = CropDataset(val_csv,   transform=get_transforms("val"))

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=nw, pin_memory=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=nw, pin_memory=True,
        ),
    }

    if test_csv is not None:
        test_ds = CropDataset(test_csv, transform=get_transforms("val"))
        loaders["test"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=nw, pin_memory=True,
        )

    return loaders


# ---------------------------------------------------------------------------
# Param summary
# ---------------------------------------------------------------------------
def print_model_summary(model: nn.Module) -> None:
    """
    Print total and trainable parameter counts for *model*.

    Parameters
    ----------
    model : Any ``torch.nn.Module``.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print("\n" + "=" * 50)
    print("  Model Parameter Summary")
    print("=" * 50)
    print(f"  Total params     : {total:>12,}")
    print(f"  Trainable params : {trainable:>12,}  ({100*trainable/total:.1f}%)")
    print(f"  Frozen params    : {frozen:>12,}  ({100*frozen/total:.1f}%)")
    print("=" * 50)

    print("\n  Layer-wise trainability:")
    print(f"  {'Layer':<25} {'Trainable':>12}")
    print(f"  {'-'*40}")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable_p = sum(p.numel() for p in module.parameters() if p.requires_grad)
        status = f"{trainable_p:>10,} / {params:>10,}"
        print(f"  {name:<25} {status}")
    print()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build model
    model = build_model(pretrained=True)
    print_model_summary(model)

    # Verify forward pass
    dummy = torch.randn(4, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"  Forward pass OK — input: {tuple(dummy.shape)}  output: {tuple(out.shape)}")

    # DataLoader smoke test (uses actual CSVs if present)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    train_csv = PROJECT_ROOT / "data" / "plantvillage" / "train.csv"
    val_csv   = PROJECT_ROOT / "data" / "plantvillage" / "val.csv"
    test_csv  = PROJECT_ROOT / "data" / "plantvillage" / "test.csv"

    if train_csv.exists():
        loaders = get_dataloaders(train_csv, val_csv, test_csv)
        for split, dl in loaders.items():
            imgs, labels = next(iter(dl))
            print(f"  {split:<6} batch — images: {tuple(imgs.shape)}  labels: {tuple(labels.shape)}")
    else:
        print("\n  [INFO] No plantvillage CSVs found — skipping DataLoader smoke test.")

    print("\n✓ cnn_model.py OK\n")
