"""
src/fusion/sensor_fetcher.py
==============================
Fetch 90 days of daily weather and hourly soil-moisture data from the
Open-Meteo **Archive API** and persist a clean sensor CSV.

Why the Archive API?
--------------------
The standard ``/v1/forecast`` endpoint only returns *future* data.
Historical values require ``https://archive-api.open-meteo.com/v1/archive``,
which stores ERA5-Land reanalysis data up to ~5 days before today.

Location: Nashik district, Maharashtra, India  (lat=21.35, lon=74.88)

Outputs
-------
  data/sensor/sensor_data.csv       – 90-row daily sensor table
  outputs/metrics/sensor_trends.png – 5-panel trend plot

Usage
-----
    python -m src.fusion.sensor_fetcher
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENSOR_DIR   = PROJECT_ROOT / "data"    / "sensor"
SENSOR_CSV   = SENSOR_DIR   / "sensor_data.csv"
METRICS_DIR  = PROJECT_ROOT / "outputs" / "metrics"
PLOT_PATH    = METRICS_DIR  / "sensor_trends.png"

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE     = 21.35
LONGITUDE    = 74.88
TIMEZONE     = "Asia/Kolkata"
LOOKBACK_DAYS = 90


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------
def fetch_weather(
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    lookback: int = LOOKBACK_DAYS,
) -> pd.DataFrame | None:
    """
    Request 90 days of daily weather + hourly soil moisture from the
    Open-Meteo Archive API.

    Parameters
    ----------
    lat      : Latitude of the field centre.
    lon      : Longitude of the field centre.
    lookback : Number of past days to retrieve.

    Returns
    -------
    DataFrame with columns
    ``[date, temperature, humidity, precipitation,
       evapotranspiration, soil_moisture]``
    or ``None`` if the request fails.
    """
    end_dt   = datetime.now() - timedelta(days=6)   # archive has ~5-day lag
    start_dt = end_dt - timedelta(days=lookback - 1)
    end   = end_dt.strftime("%Y-%m-%d")
    start = start_dt.strftime("%Y-%m-%d")

    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "et0_fao_evapotranspiration",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
        ]),
        "hourly":   "soil_moisture_0_to_7cm",
        "timezone": TIMEZONE,
    }

    try:
        print(f"  Requesting archive data ({start} → {end}) …")
        r = requests.get(ARCHIVE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"  API error: {exc}")
        return None

    # ── Daily variables ────────────────────────────────────────────────
    daily = pd.DataFrame(data["daily"])
    daily.rename(columns={
        "time":                        "date",
        "temperature_2m_max":          "temp_max",
        "temperature_2m_min":          "temp_min",
        "precipitation_sum":           "precipitation",
        "et0_fao_evapotranspiration":  "evapotranspiration",
        "relative_humidity_2m_max":    "humidity_max",
        "relative_humidity_2m_min":    "humidity_min",
    }, inplace=True)

    daily["temperature"] = (daily["temp_max"] + daily["temp_min"]) / 2
    daily["humidity"]    = (daily["humidity_max"] + daily["humidity_min"]) / 2

    # ── Hourly → daily soil moisture ──────────────────────────────────
    hourly = pd.DataFrame(data["hourly"])
    hourly["date"] = pd.to_datetime(hourly["time"]).dt.date.astype(str)
    soil_daily = (
        hourly.groupby("date")["soil_moisture_0_to_7cm"]
        .mean()
        .reset_index()
    )
    soil_daily.columns = ["date", "soil_moisture"]

    # ── Merge & select ────────────────────────────────────────────────
    df = daily.merge(soil_daily, on="date", how="left")
    df = df[[
        "date", "temperature", "humidity",
        "precipitation", "evapotranspiration", "soil_moisture",
    ]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

    return df


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------
def synthetic_weather(
    n: int = LOOKBACK_DAYS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate plausible synthetic sensor data for Nashik, Maharashtra.

    Used when the API call fails (e.g. no internet or rate-limit).

    Parameters
    ----------
    n    : Number of daily rows to generate.
    seed : NumPy random seed.

    Returns
    -------
    DataFrame with the same schema as :func:`fetch_weather`.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(end=datetime.now(), periods=n).strftime("%Y-%m-%d")

    return pd.DataFrame({
        "date":               dates,
        "temperature":        rng.uniform(20, 40, n),
        "humidity":           rng.uniform(40, 90, n),
        "precipitation":      np.abs(rng.normal(5, 10, n)),
        "evapotranspiration": rng.uniform(2, 8, n),
        "soil_moisture":      rng.uniform(0.10, 0.45, n),
    })


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_trends(df: pd.DataFrame, save_path: Path = PLOT_PATH) -> None:
    """
    Save a 5-panel time-series plot of all sensor variables.

    Parameters
    ----------
    df        : Daily sensor DataFrame from :func:`fetch_weather` or
                :func:`synthetic_weather`.
    save_path : Destination PNG path.
    """
    variables = [
        ("temperature",        "Temperature (°C)",        "#e74c3c"),
        ("humidity",           "Humidity (%)",             "#3498db"),
        ("precipitation",      "Precipitation (mm)",       "#2ecc71"),
        ("evapotranspiration", "Evapotranspiration (mm)", "#f39c12"),
        ("soil_moisture",      "Soil Moisture (m³/m³)",   "#8e44ad"),
    ]

    fig, axes = plt.subplots(len(variables), 1, figsize=(14, 3 * len(variables)),
                             sharex=True)
    fig.suptitle(
        f"90-Day Sensor Trends — Nashik, Maharashtra\n"
        f"({df['date'].iloc[0]} → {df['date'].iloc[-1]})",
        fontsize=13, fontweight="bold",
    )

    x = range(len(df))
    for ax, (col, ylabel, color) in zip(axes, variables):
        ax.plot(x, df[col].values, color=color, linewidth=1.5)
        ax.fill_between(x, df[col].values, alpha=0.15, color=color)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)

    # x-tick labels every ~15 days
    step = max(1, len(df) // 6)
    axes[-1].set_xticks(list(x)[::step])
    axes[-1].set_xticklabels(df["date"].iloc[::step].tolist(),
                              rotation=30, ha="right", fontsize=7)
    axes[-1].set_xlabel("Date", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Fetch → (fallback) → save CSV → plot → print preview."""
    SENSOR_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Sensor Fetcher  (Open-Meteo Archive API)")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Attempt live API fetch
    # ------------------------------------------------------------------
    df = fetch_weather()

    if df is None or df.empty:
        print("\n  WARNING: Using synthetic sensor data (API unavailable).")
        df = synthetic_weather()
    else:
        print(f"  ✓ Retrieved {len(df)} days of real sensor data.")

    # ------------------------------------------------------------------
    # 2. Save CSV
    # ------------------------------------------------------------------
    df.to_csv(SENSOR_CSV, index=False)
    print(f"  CSV saved → {SENSOR_CSV}")

    # ------------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------------
    print("  Plotting sensor trends …")
    plot_trends(df)

    # ------------------------------------------------------------------
    # 4. Preview
    # ------------------------------------------------------------------
    print("\n  First 5 rows of saved CSV:")
    print(df.head().to_string(index=False))

    print("\n" + "=" * 55)
    print(f"  Rows          : {len(df)}")
    print(f"  Date range    : {df['date'].iloc[0]}  →  {df['date'].iloc[-1]}")
    for col in ["temperature", "humidity", "precipitation",
                "evapotranspiration", "soil_moisture"]:
        print(f"  {col:22s}: "
              f"min={df[col].min():.3f}  max={df[col].max():.3f}  "
              f"mean={df[col].mean():.3f}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
