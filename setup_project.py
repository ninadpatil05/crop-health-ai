"""
setup_project.py — Project scaffold for AI-powered Crop Health Monitoring.
Run once to create all required directories.
"""

import os
from pathlib import Path

# ── Root of the project (same directory as this script) ──────────────────────
ROOT = Path(__file__).parent.resolve()

DIRS = [
    # Sentinel-2 satellite data
    "data/sentinel2/raw",
    "data/sentinel2/processed",
    "data/sentinel2/patches",
    # Other datasets
    "data/plantvillage",
    "data/landsat",
    "data/indian_pines",
    "data/sensor",
    "data/combined",
    # Saved model weights
    "models/cnn",
    "models/lstm",
    "models/fusion",
    # Source-code sub-packages
    "src/preprocessing",
    "src/indices",
    "src/training",
    "src/inference",
    "src/fusion",
    "src/alerts",
    # Streamlit dashboard
    "dashboard",
    # Outputs
    "outputs/maps",
    "outputs/reports",
    "outputs/metrics",
    "outputs/alerts",
    # Misc
    "notebooks",
    "tests",
    "docs",
    "config",
]


def main() -> None:
    for rel_dir in DIRS:
        target: Path = ROOT / rel_dir
        os.makedirs(target, exist_ok=True)
        print(f"  [OK] {target.relative_to(ROOT)}")

    print("\nSetup complete! All folders created.")


if __name__ == "__main__":
    main()
