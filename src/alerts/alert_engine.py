"""
src/alerts/alert_engine.py
============================
Rule-based crop-health alert engine.

Reads per-zone risk scores (from ``outputs/maps/risk_scores.json``),
classifies each zone into one of four alert levels, generates a
recommended action, and persists the results as JSON.

Alert levels
------------
  0–30   CLEAR    – no action required
  31–60  WATCH    – monitor field conditions
  61–80  WARNING  – inspect within 72 h
  81–100 CRITICAL – act immediately

Outputs
-------
  outputs/alerts/alerts_YYYYMMDD.json  – all zone alerts
  outputs/alerts/active_alerts.json    – WARNING + CRITICAL only

Usage
-----
    python -m src.alerts.alert_engine

Streamlit integration
---------------------
    from src.alerts.alert_engine import trigger_alerts
    result = trigger_alerts(risk_scores_dict)
    # result = {'summary': {...}, 'active': [...], 'all': [...]}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT      = Path(__file__).resolve().parents[2]
RISK_JSON         = PROJECT_ROOT / "outputs" / "maps"   / "risk_scores.json"
ALERTS_DIR        = PROJECT_ROOT / "outputs" / "alerts"
ACTIVE_ALERT_PATH = ALERTS_DIR / "active_alerts.json"

# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------
LEVEL_THRESHOLDS = [
    (81, "CRITICAL"),
    (61, "WARNING"),
    (31, "WATCH"),
    (0,  "CLEAR"),
]


# ---------------------------------------------------------------------------
# Core pure functions (also used by unit tests)
# ---------------------------------------------------------------------------

def get_level(risk_score: float) -> str:
    """
    Map a numeric risk score (0–100) to an alert level string.

    Parameters
    ----------
    risk_score : Float in [0, 100].

    Returns
    -------
    One of ``'CRITICAL'``, ``'WARNING'``, ``'WATCH'``, ``'CLEAR'``.

    Examples
    --------
    >>> get_level(85)
    'CRITICAL'
    >>> get_level(20)
    'CLEAR'
    """
    for threshold, level in LEVEL_THRESHOLDS:
        if risk_score >= threshold:
            return level
    return "CLEAR"


def get_action(
    cnn_class:   str,
    risk_score:  float,
    lstm_ndvi:   float,
    alert_level: str | None = None,
) -> str:
    """
    Derive the recommended field action from classification and risk data.

    Rules are evaluated in priority order:

    1. Fungal Disease + risk > 60  → apply fungicide
    2. Pest Damage    + risk > 60  → deploy pesticide
    3. Bacterial      + risk > 60  → remove material + bactericide
    4. Low NDVI forecast + risk > 50 → irrigate
    5. WARNING / WATCH             → schedule inspection
    6. Otherwise                   → no action

    Parameters
    ----------
    cnn_class   : Predicted crop disease class (string).
    risk_score  : Composite risk score in [0, 100].
    lstm_ndvi   : LSTM-forecast NDVI value (next step).
    alert_level : Pre-computed alert level; computed from *risk_score* if None.

    Returns
    -------
    Human-readable action string.
    """
    if alert_level is None:
        alert_level = get_level(risk_score)

    if cnn_class == "Fungal Disease" and risk_score > 60:
        return "Apply preventive fungicide within 72 hours"
    elif cnn_class == "Pest Damage" and risk_score > 60:
        return "Deploy pesticide spray, inspect field within 48 hours"
    elif cnn_class == "Bacterial" and risk_score > 60:
        return "Remove infected material, apply copper-based bactericide"
    elif lstm_ndvi < 0.35 and risk_score > 50:
        return "Initiate irrigation — NDVI forecast below critical threshold"
    elif alert_level in ("WARNING", "WATCH"):
        return "Schedule field inspection within 7 days"
    else:
        return "No action required"


# ---------------------------------------------------------------------------
# Alert builder
# ---------------------------------------------------------------------------

def build_alerts(risk_scores: dict[str, Any]) -> list[dict]:
    """
    Convert a risk-scores dict into a list of alert records.

    Parameters
    ----------
    risk_scores : Mapping of zone_id → ``{risk_score, cnn_class,
                  confidence, lstm_ndvi}``.

    Returns
    -------
    List of alert dicts, each containing:
    ``zone_id, risk_score, cnn_class, confidence, lstm_ndvi,
      alert_level, action, timestamp``.
    """
    ts    = datetime.now().isoformat(timespec="seconds")
    alerts: list[dict] = []

    for zone_id, info in risk_scores.items():
        risk       = float(info.get("risk_score",  0))
        cnn_class  = str(info.get("cnn_class",    "Unknown"))
        confidence = float(info.get("confidence", 0.0))
        lstm_ndvi  = float(info.get("lstm_ndvi",  0.5))

        level  = get_level(risk)
        action = get_action(cnn_class, risk, lstm_ndvi, alert_level=level)

        alerts.append({
            "zone_id":     zone_id,
            "risk_score":  risk,
            "cnn_class":   cnn_class,
            "confidence":  confidence,
            "lstm_ndvi":   lstm_ndvi,
            "alert_level": level,
            "action":      action,
            "timestamp":   ts,
        })

    return alerts


def summarise(alerts: list[dict]) -> dict[str, int]:
    """
    Count alerts per level.

    Returns
    -------
    ``{'CRITICAL': N, 'WARNING': N, 'WATCH': N, 'CLEAR': N}``
    """
    counts: dict[str, int] = {"CRITICAL": 0, "WARNING": 0, "WATCH": 0, "CLEAR": 0}
    for a in alerts:
        counts[a["alert_level"]] = counts.get(a["alert_level"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Streamlit-compatible API
# ---------------------------------------------------------------------------

def trigger_alerts(risk_scores_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Public entry point for Streamlit / API integration.

    Parameters
    ----------
    risk_scores_dict : Same structure as ``outputs/maps/risk_scores.json``.

    Returns
    -------
    dict with keys:
      ``summary`` – level counts
      ``active``  – WARNING + CRITICAL alerts
      ``all``     – every alert
    """
    alerts = build_alerts(risk_scores_dict)
    active = [a for a in alerts if a["alert_level"] in ("WARNING", "CRITICAL")]
    return {
        "summary": summarise(alerts),
        "active":  active,
        "all":     alerts,
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load or mock risk scores
    # ------------------------------------------------------------------
    if not RISK_JSON.exists():
        print("  risk_scores.json not found — generating mock data …")
        mock = {
            str(i): {
                "risk_score": i,
                "cnn_class":  "Fungal Disease",
                "confidence": 0.8,
                "lstm_ndvi":  0.3,
            }
            for i in range(0, 101, 5)
        }
        RISK_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(RISK_JSON, "w") as f:
            json.dump(mock, f)
        risk_scores = mock
        print(f"  Mock data written → {RISK_JSON}")
    else:
        with open(RISK_JSON) as f:
            risk_scores = json.load(f)
        print(f"  Loaded risk scores → {RISK_JSON}  ({len(risk_scores)} zones)")

    # ------------------------------------------------------------------
    # 2. Build alerts
    # ------------------------------------------------------------------
    result = trigger_alerts(risk_scores)
    alerts = result["all"]
    active = result["active"]
    summary = result["summary"]

    # ------------------------------------------------------------------
    # 3. Save outputs
    # ------------------------------------------------------------------
    dated_path = ALERTS_DIR / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
    _save_json(alerts, dated_path)
    _save_json(active, ACTIVE_ALERT_PATH)
    print(f"  All alerts   → {dated_path}  ({len(alerts)} zones)")
    print(f"  Active alerts→ {ACTIVE_ALERT_PATH}  ({len(active)} zones)")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  Alert Summary")
    print("=" * 50)
    for level in ("CRITICAL", "WARNING", "WATCH", "CLEAR"):
        bar = "█" * summary[level]
        print(f"  {level:<10}: {summary[level]:>4}  {bar}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 5. Print active alerts
    # ------------------------------------------------------------------
    if active:
        print(f"\n  ── Active Alerts ({len(active)}) ──────────────────────")
        for a in sorted(active, key=lambda x: -x["risk_score"])[:10]:
            print(f"  [{a['alert_level']:<8}] Zone {a['zone_id']:>4} "
                  f"| risk={a['risk_score']:5.1f} "
                  f"| {a['cnn_class']:<16} | {a['action']}")
        if len(active) > 10:
            print(f"  … and {len(active)-10} more")
    else:
        print("\n  No active alerts.")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
