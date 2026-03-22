"""
config/settings.py — Central configuration for the Crop Health AI project.
All path constants are expressed as POSIX-style strings (relative to project root).
Use pathlib.Path(SENTINEL_RAW) etc. in application code for cross-platform safety.
"""

# ── Data directories ───────────────────────────────────────────────────────────
SENTINEL_RAW   = 'data/sentinel2/raw'
SENTINEL_PROC  = 'data/sentinel2/processed'
PLANTVILLAGE   = 'data/plantvillage'
LANDSAT_DIR    = 'data/landsat'
INDIAN_PINES   = 'data/indian_pines'
SENSOR_DIR     = 'data/sensor'

# ── Model directories ──────────────────────────────────────────────────────────
MODELS_CNN     = 'models/cnn'
MODELS_LSTM    = 'models/lstm'
MODELS_FUSION  = 'models/fusion'

# ── Output directories ─────────────────────────────────────────────────────────
OUT_MAPS       = 'outputs/maps'
OUT_METRICS    = 'outputs/metrics'
OUT_REPORTS    = 'outputs/reports'
OUT_ALERTS     = 'outputs/alerts'

# ── Field location (WGS-84 decimal degrees) ───────────────────────────────────
FIELD_LAT      = 21.35
FIELD_LON      = 74.88

# ── NDVI health thresholds ────────────────────────────────────────────────────
NDVI_HEALTHY_MIN = 0.4    # NDVI >= this → healthy crop
NDVI_STRESS_MAX  = 0.35   # NDVI <= this → stressed crop

# ── Disease-risk score thresholds (0–100 scale) ───────────────────────────────
RISK_HIGH        = 60     # score >= 60 → high-risk alert
RISK_CRITICAL    = 80     # score >= 80 → critical alert
