"""
csv_exporter.py
---------------
Export per-zone field report to CSV by joining:
  - Spectral index maps (NDVI, NDRE, SAVI, EVI, NDWI, BSI)
  - CNN / LSTM risk scores
  - Active alerts
  - Latest sensor row
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


def export_csv(output_path: str | None = None) -> pd.DataFrame:
    """Generate a 100-zone field report CSV and return the DataFrame."""

    # ------------------------------------------------------------------ #
    #  Output path                                                         #
    # ------------------------------------------------------------------ #
    date_str = datetime.now().strftime("%Y%m%d")
    if output_path is None:
        output_path = f"outputs/reports/field_data_{date_str}.csv"

    # ------------------------------------------------------------------ #
    #  Load spectral index maps (graceful fallback to zeros)              #
    # ------------------------------------------------------------------ #
    def load_index(name: str) -> np.ndarray:
        path = Path(f"outputs/maps/{name}.npy")
        if path.exists():
            return np.load(path)
        print(f"  [WARN] {path} not found – using zeros(512,512)")
        return np.zeros((512, 512))

    ndvi = load_index("ndvi")
    ndre = load_index("ndre")
    savi = load_index("savi")
    evi  = load_index("evi")
    ndwi = load_index("ndwi")
    bsi  = load_index("bsi")

    # ------------------------------------------------------------------ #
    #  Zone statistics – 10×10 grid → 100 zones                          #
    # ------------------------------------------------------------------ #
    H, W = ndvi.shape
    zone_stats = {}
    for i in range(100):
        row = i // 10
        col = i % 10
        r0, r1 = row * H // 10, (row + 1) * H // 10
        c0, c1 = col * W // 10, (col + 1) * W // 10

        zone_stats[i] = {
            "zone_id":   i,
            "mean_ndvi": ndvi[r0:r1, c0:c1].mean(),
            "mean_ndre": ndre[r0:r1, c0:c1].mean(),
            "mean_savi": savi[r0:r1, c0:c1].mean(),
            "mean_evi":  evi[r0:r1, c0:c1].mean(),
            "mean_ndwi": ndwi[r0:r1, c0:c1].mean(),
            "mean_bsi":  bsi[r0:r1, c0:c1].mean(),
        }

    df = pd.DataFrame.from_dict(zone_stats, orient="index")

    # ------------------------------------------------------------------ #
    #  Join risk scores                                                    #
    # ------------------------------------------------------------------ #
    risk_path = Path("outputs/maps/risk_scores.json")
    if risk_path.exists():
        with open(risk_path) as f:
            risk_raw = json.load(f)
        # Keys are string zone IDs; normalise to int
        risk = {int(k): v for k, v in risk_raw.items()}
    else:
        print("  [WARN] risk_scores.json not found – risk columns will be empty")
        risk = {}

    df["cnn_class"]          = df["zone_id"].map(lambda z: risk.get(z, {}).get("cnn_class",  "Unknown"))
    df["confidence_pct"]     = df["zone_id"].map(lambda z: risk.get(z, {}).get("confidence", 0.0)) * 100
    df["risk_score"]         = df["zone_id"].map(lambda z: risk.get(z, {}).get("risk_score", 0.0))
    df["lstm_ndvi_forecast"] = df["zone_id"].map(lambda z: risk.get(z, {}).get("lstm_ndvi",  np.nan))

    # ------------------------------------------------------------------ #
    #  Join alerts                                                         #
    # ------------------------------------------------------------------ #
    alerts_path = Path("outputs/alerts/active_alerts.json")
    if alerts_path.exists():
        with open(alerts_path) as f:
            alerts_raw = json.load(f)
        # Build lookup: zone_id (int) → {alert_level, recommended_action}
        alert_lookup: dict[int, dict] = {}
        for a in alerts_raw:
            zid = int(a.get("zone_id", -1))
            alert_lookup[zid] = {
                "alert_level":        a.get("alert_level", "CLEAR"),
                "recommended_action": a.get("action", "No action required"),
            }
    else:
        print("  [WARN] active_alerts.json not found – all zones set to CLEAR")
        alert_lookup = {}

    df["alert_level"] = df["zone_id"].map(
        lambda z: alert_lookup.get(z, {}).get("alert_level", "CLEAR")
    )
    df["recommended_action"] = df["zone_id"].map(
        lambda z: alert_lookup.get(z, {}).get("recommended_action", "No action required")
    )

    # ------------------------------------------------------------------ #
    #  Join sensor data – broadcast latest row to all zones               #
    # ------------------------------------------------------------------ #
    sensor_path = Path("data/sensor/sensor_data.csv")
    if sensor_path.exists():
        sensor = pd.read_csv(sensor_path).iloc[-1]
        df["soil_moisture"]  = sensor.get("soil_moisture", np.nan)
        df["temperature_c"]  = sensor.get("temperature",  np.nan)
        df["humidity_pct"]   = sensor.get("humidity",     np.nan)
    else:
        print("  [WARN] sensor_data.csv not found – sensor columns will be NaN")
        df["soil_moisture"] = np.nan
        df["temperature_c"] = np.nan
        df["humidity_pct"]  = np.nan

    # ------------------------------------------------------------------ #
    #  Add report date and enforce column order                           #
    # ------------------------------------------------------------------ #
    df["report_date"] = datetime.now().strftime("%Y-%m-%d")

    columns = [
        "zone_id",
        "mean_ndvi", "mean_ndre", "mean_savi", "mean_evi", "mean_ndwi", "mean_bsi",
        "cnn_class", "confidence_pct", "risk_score", "alert_level", "lstm_ndvi_forecast",
        "soil_moisture", "temperature_c", "humidity_pct",
        "recommended_action", "report_date",
    ]
    df = df[columns]

    # ------------------------------------------------------------------ #
    #  Round & export                                                      #
    # ------------------------------------------------------------------ #
    df = df.round(4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} zones to {output_path}")
    return df


if __name__ == "__main__":
    export_csv()
