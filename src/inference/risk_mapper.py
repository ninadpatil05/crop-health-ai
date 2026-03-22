"""
src/inference/risk_mapper.py
==============================
Fuse CNN classification, LSTM NDVI forecast, and real-time sensor data
into per-zone composite risk scores, then visualise them as both a static
PNG heatmap and an interactive Folium HTML map.

Risk scoring
------------
  base  = 0 (Healthy) or confidence × 50 (any disease class)
  +30   if LSTM-forecast NDVI < 0.35  (vegetation stress)
  +10   if soil_moisture  < 0.20      (drought stress)
  +10   if humidity       > 85        (disease-prone conditions)
  risk_score = min(100, base)  [integer]

Outputs
-------
  outputs/maps/risk_scores.json  – per-zone risk dict (also consumed by alert_engine)
  outputs/maps/risk_map.png      – static 10×10 heatmap
  outputs/maps/risk_map.html     – interactive Folium map

Usage
-----
    python -m src.inference.risk_mapper
"""

from __future__ import annotations

import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.training.lstm_model import NDVIForecaster

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
CNN_PREDS_PATH  = PROJECT_ROOT / "outputs" / "metrics" / "cnn_predictions.json"
FUSED_NPZ       = PROJECT_ROOT / "data"    / "sentinel2" / "timeseries_fused.npz"
LSTM_MODEL_PATH = PROJECT_ROOT / "models"  / "fusion"   / "lstm_fused.pt"
SENSOR_CSV      = PROJECT_ROOT / "data"    / "sensor"   / "sensor_data.csv"
MAPS_DIR        = PROJECT_ROOT / "outputs" / "maps"
RISK_JSON       = MAPS_DIR / "risk_scores.json"
HEATMAP_PNG     = MAPS_DIR / "risk_map.png"
FOLIUM_HTML     = MAPS_DIR / "risk_map.html"

N_ZONES         = 100
GRID_SIZE       = 10
LAT, LON        = 21.35, 74.88   # Field centre – Nashik, Maharashtra

CNN_CLASSES = ["Healthy", "Fungal Disease", "Bacterial", "Pest Damage", "Stress"]


# ---------------------------------------------------------------------------
# 1. CNN predictions
# ---------------------------------------------------------------------------

def load_cnn_predictions(path: Path) -> dict[str, dict]:
    """
    Load CNN predictions from JSON.  Falls back to mock data if file is absent.

    Returns
    -------
    dict mapping zone_id (str) → ``{class, confidence}``.
    """
    if path.exists():
        with open(path) as f:
            raw = json.load(f)
        # raw keys are image paths → aggregate by zone index (round-robin)
        items = list(raw.items())
        preds: dict[str, dict] = {}
        for zone in range(N_ZONES):
            img_path, info = items[zone % len(items)]
            preds[str(zone)] = {
                "class":      info.get("class", "Healthy"),
                "confidence": float(info.get("confidence", 0.5)),
            }
        print(f"  CNN predictions  : loaded from {path}  ({len(raw)} images → {N_ZONES} zones)")
        return preds
    else:
        print("  CNN predictions  : file missing — generating mock data …")
        rng = random.Random(42)
        return {
            str(z): {
                "class":      rng.choice(CNN_CLASSES),
                "confidence": round(rng.uniform(0.5, 0.99), 3),
            }
            for z in range(N_ZONES)
        }


# ---------------------------------------------------------------------------
# 2. LSTM predictions
# ---------------------------------------------------------------------------

def load_lstm_predictions(
    model_path: Path,
    fused_npz:  Path,
    device:     torch.device,
) -> dict[int, float]:
    """
    Run the fused LSTM on the last sequence of each zone and return the
    predicted NDVI for the next time step.

    Returns
    -------
    dict mapping zone_index (int) → predicted NDVI (float).
    """
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model = NDVIForecaster(input_size=int(ckpt.get("input_size", 6)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    data = np.load(str(fused_npz))
    X    = data["X_sequences"]          # (N, 8, 6)
    N    = len(X)
    seqs_per_zone = N // N_ZONES

    preds: dict[int, float] = {}
    with torch.no_grad():
        for zone in range(N_ZONES):
            # Use the last sequence belonging to this zone
            idx = min((zone + 1) * seqs_per_zone - 1, N - 1)
            x_in = torch.FloatTensor(X[idx : idx + 1]).to(device)
            y_hat = model(x_in).cpu().numpy().flatten()
            preds[zone] = float(y_hat[0])

    print(f"  LSTM predictions : {N_ZONES} zones  "
          f"(NDVI range [{min(preds.values()):.3f}, {max(preds.values()):.3f}])")
    return preds


# ---------------------------------------------------------------------------
# 3. Sensor features
# ---------------------------------------------------------------------------

def load_sensor(path: Path) -> tuple[float, float]:
    """
    Return the most recent soil_moisture and humidity values.

    Returns
    -------
    (soil_moisture, humidity)
    """
    sensor        = pd.read_csv(str(path)).iloc[-1]
    soil_moisture = float(sensor["soil_moisture"])
    humidity      = float(sensor["humidity"])
    print(f"  Sensor (latest)  : soil_moisture={soil_moisture:.3f}  humidity={humidity:.1f}%")
    return soil_moisture, humidity


# ---------------------------------------------------------------------------
# 4. Risk scoring
# ---------------------------------------------------------------------------

def compute_risk_scores(
    cnn_preds:    dict[str, dict],
    lstm_preds:   dict[int, float],
    soil_moisture: float,
    humidity:      float,
) -> dict[str, dict]:
    """
    Compute a composite risk score [0, 100] for each of the 100 zones.

    Parameters
    ----------
    cnn_preds     : Zone-level CNN class + confidence.
    lstm_preds    : Zone-level LSTM-forecast NDVI.
    soil_moisture : Latest soil moisture reading (m³/m³).
    humidity      : Latest relative humidity reading (%).

    Returns
    -------
    dict zone_id → ``{risk_score, cnn_class, confidence, lstm_ndvi,
                       soil_moisture, humidity}``.
    """
    risk_scores: dict[str, dict] = {}

    for zone in range(N_ZONES):
        cnn_info   = cnn_preds.get(str(zone), {"class": "Healthy", "confidence": 0.5})
        cnn_class  = cnn_info["class"]
        confidence = float(cnn_info["confidence"])
        lstm_ndvi  = lstm_preds.get(zone, 0.5)

        # Base score from CNN
        base = 0.0 if cnn_class == "Healthy" else confidence * 50

        # Additive penalty terms
        if lstm_ndvi       < 0.35:   base += 30
        if soil_moisture   < 0.20:   base += 10
        if humidity        > 85:     base += 10

        risk_score = int(min(100, base))

        risk_scores[str(zone)] = {
            "risk_score":    risk_score,
            "cnn_class":     cnn_class,
            "confidence":    round(confidence, 4),
            "lstm_ndvi":     round(lstm_ndvi, 4),
            "soil_moisture": round(soil_moisture, 4),
            "humidity":      round(humidity, 2),
        }

    return risk_scores


# ---------------------------------------------------------------------------
# 5. PNG heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(risk_scores: dict[str, dict], save_path: Path) -> None:
    """
    Save a 10×10 colour-coded risk heatmap as PNG.

    Parameters
    ----------
    risk_scores : Output of :func:`compute_risk_scores`.
    save_path   : Destination PNG path.
    """
    grid = np.array(
        [risk_scores[str(i)]["risk_score"] for i in range(N_ZONES)],
        dtype=float,
    ).reshape(GRID_SIZE, GRID_SIZE)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Risk Score (0–100)")

    ax.set_title("Field Risk Map (10×10 Zone Grid)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Column (East →)")
    ax.set_ylabel("Row (North ↑)")

    # Zone labels
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            zone  = r * GRID_SIZE + c
            score = int(grid[r, c])
            color = "white" if score > 60 else "black"
            ax.text(c, r, str(score), ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved    : {save_path}")


# ---------------------------------------------------------------------------
# 6. Folium map
# ---------------------------------------------------------------------------

def build_folium_map(risk_scores: dict[str, dict], save_path: Path) -> None:
    """
    Build an interactive Folium map with colour-coded zone markers.

    Risk colours
    ------------
    risk < 31  → green  (#22C55E)
    risk < 61  → amber  (#F59E0B)
    risk ≥ 61  → red    (#EF4444)

    Parameters
    ----------
    risk_scores : Output of :func:`compute_risk_scores`.
    save_path   : Destination HTML path.
    """
    m = folium.Map(location=[LAT, LON], zoom_start=14)

    # Lay out 100 zones on a ~1 km grid (≈0.009° per 1 km)
    cell_deg = 0.001
    for zone in range(N_ZONES):
        row  = zone // GRID_SIZE
        col  = zone  % GRID_SIZE
        zlat = LAT + row * cell_deg
        zlon = LON + col * cell_deg

        info  = risk_scores[str(zone)]
        score = info["risk_score"]
        color = "#22C55E" if score < 31 else ("#F59E0B" if score < 61 else "#EF4444")

        folium.CircleMarker(
            location=[zlat, zlon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>Zone {zone}</b><br>"
                f"Risk: <b>{score}</b><br>"
                f"Class: {info['cnn_class']}<br>"
                f"Confidence: {info['confidence']:.1%}<br>"
                f"LSTM NDVI: {info['lstm_ndvi']:.3f}",
                max_width=200,
            ),
            tooltip=f"Zone {zone} | Risk {score}",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;'>
      <b>Risk Level</b><br>
      <span style='color:#22C55E'>●</span> CLEAR (0–30)<br>
      <span style='color:#F59E0B'>●</span> WATCH/WARNING (31–80)<br>
      <span style='color:#EF4444'>●</span> CRITICAL (81–100)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(str(save_path))
    print(f"  Folium map saved : {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Risk Mapper — CNN + LSTM + Sensor Fusion")
    print("=" * 55)

    # Device
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"\n  Device           : {device}")

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print()
    cnn_preds            = load_cnn_predictions(CNN_PREDS_PATH)
    lstm_preds           = load_lstm_predictions(LSTM_MODEL_PATH, FUSED_NPZ, device)
    soil_moisture, humidity = load_sensor(SENSOR_CSV)

    # ------------------------------------------------------------------
    # 2. Compute risk scores
    # ------------------------------------------------------------------
    print("\n  Computing risk scores …")
    risk_scores = compute_risk_scores(cnn_preds, lstm_preds, soil_moisture, humidity)

    # ------------------------------------------------------------------
    # 3. Save JSON
    # ------------------------------------------------------------------
    with open(RISK_JSON, "w") as f:
        json.dump(risk_scores, f, indent=2)
    print(f"  Risk JSON saved  : {RISK_JSON}  ({len(risk_scores)} entries)")

    # Verify 100 entries
    assert len(risk_scores) == N_ZONES, f"Expected {N_ZONES} zones, got {len(risk_scores)}"
    print(f"  ✓ risk_scores.json contains {len(risk_scores)} entries")

    # ------------------------------------------------------------------
    # 4. Heatmap
    # ------------------------------------------------------------------
    plot_heatmap(risk_scores, HEATMAP_PNG)

    # ------------------------------------------------------------------
    # 5. Folium map
    # ------------------------------------------------------------------
    build_folium_map(risk_scores, FOLIUM_HTML)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    scores = [v["risk_score"] for v in risk_scores.values()]
    high   = sum(1 for s in scores if s  > 60)
    medium = sum(1 for s in scores if 31 <= s <= 60)
    low    = sum(1 for s in scores if s  < 31)

    crit  = sum(1 for s in scores if s  > 80)
    warn  = sum(1 for s in scores if 61 <= s <= 80)
    watch = sum(1 for s in scores if 31 <= s <= 60)
    clear = sum(1 for s in scores if s  < 31)

    print("\n" + "=" * 55)
    print("  Risk Summary")
    print("=" * 55)
    print(f"  Total zones      : {N_ZONES}")
    print(f"  High risk (>60)  : {high}  zones")
    print(f"  Medium  (31-60)  : {medium}  zones")
    print(f"  Low     (<31)    : {low}  zones")
    print(f"  ─────────────────────────────")
    print(f"  CRITICAL (81-100): {crit}")
    print(f"  WARNING  (61-80) : {warn}")
    print(f"  WATCH    (31-60) : {watch}")
    print(f"  CLEAR    ( 0-30) : {clear}")
    print(f"  Mean risk score  : {np.mean(scores):.1f}")
    print(f"  Max risk score   : {np.max(scores)}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
