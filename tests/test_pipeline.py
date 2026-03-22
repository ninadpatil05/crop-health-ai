"""
tests/test_pipeline.py
----------------------
End-to-end pipeline smoke tests.
Run with: pytest tests/test_pipeline.py -v
"""

import json
import numpy as np
import pytest
import torch
from pathlib import Path


# ------------------------------------------------------------------ #
# TEST 1 — Sentinel-2 processed stack                                 #
# ------------------------------------------------------------------ #
def test_sentinel_stack():
    path = Path("data/sentinel2/processed/s2_stack.npy")
    assert path.exists(), "Run P1-02 first"

    stack = np.load(path)

    assert stack.ndim == 3, f"Expected 3D array, got {stack.ndim}D"
    assert stack.shape[2] == 6, f"Expected 6 bands, got {stack.shape[2]}"
    assert stack.min() >= 0.0, "Values below 0 found"
    assert stack.max() <= 1.0, "Values above 1 found"
    assert not np.isnan(stack).any(), "NaN values found in stack"


# ------------------------------------------------------------------ #
# TEST 2 — Vegetation / spectral indices                              #
# ------------------------------------------------------------------ #
def test_indices():
    for name in ["ndvi", "ndre", "savi", "evi", "ndwi", "bsi"]:
        path = Path(f"outputs/maps/{name}.npy")
        assert path.exists(), f"{name}.npy missing — run P1-03"

        arr = np.load(path)
        assert arr.min() >= -1.0 and arr.max() <= 1.0, f"{name} out of range"
        assert not np.isnan(arr).any(), f"NaN in {name}"


# ------------------------------------------------------------------ #
# TEST 3 — CNN forward pass                                           #
# ------------------------------------------------------------------ #
def test_cnn_inference():
    from src.training.cnn_model import CropDiseaseCNN

    ckpt_path = Path("models/cnn/best_model.pt")
    assert ckpt_path.exists(), "Run P2-02 first"

    model = CropDiseaseCNN()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(5, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (5, 5), f"Expected (5,5), got {out.shape}"

    probs = torch.softmax(out, dim=1).numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# ------------------------------------------------------------------ #
# TEST 4 — LSTM forward pass                                          #
# ------------------------------------------------------------------ #
def test_lstm_inference():
    from src.training.lstm_model import NDVIForecaster

    ckpt_path = Path("models/fusion/lstm_fused.pt")
    assert ckpt_path.exists(), "Run P3-04 first"

    model = NDVIForecaster(input_size=6)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 8, 6)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (1, 2), f"Expected (1,2), got {out.shape}"


# ------------------------------------------------------------------ #
# TEST 5 — Risk scores JSON                                           #
# ------------------------------------------------------------------ #
def test_risk_scores():
    path = Path("outputs/maps/risk_scores.json")
    assert path.exists(), "Run P4-01 first"

    scores = json.load(open(path))
    assert len(scores) == 100, f"Expected 100 zones, got {len(scores)}"

    for zone_id, info in scores.items():
        assert 0 <= info["risk_score"] <= 100, f"Zone {zone_id} risk out of range"
