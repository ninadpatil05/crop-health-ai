"""
tests/test_alert_engine.py
============================
Unit tests for src.alerts.alert_engine.

Run with:
    pytest tests/test_alert_engine.py -v
"""

import pytest
from src.alerts.alert_engine import get_level, get_action, build_alerts, summarise


# ---------------------------------------------------------------------------
# get_level tests
# ---------------------------------------------------------------------------

def test_critical():
    assert get_level(85) == "CRITICAL"

def test_critical_boundary():
    assert get_level(81) == "CRITICAL"

def test_warning():
    assert get_level(70) == "WARNING"

def test_warning_boundary():
    assert get_level(61) == "WARNING"

def test_watch():
    assert get_level(45) == "WATCH"

def test_watch_boundary():
    assert get_level(31) == "WATCH"

def test_clear():
    assert get_level(20) == "CLEAR"

def test_clear_zero():
    assert get_level(0) == "CLEAR"

def test_clear_boundary():
    assert get_level(30) == "CLEAR"


# ---------------------------------------------------------------------------
# get_action tests
# ---------------------------------------------------------------------------

def test_action_fungal():
    """Fungal Disease with high risk → fungicide."""
    action = get_action("Fungal Disease", 75, 0.4)
    assert "fungicide" in action.lower()

def test_action_pest():
    """Pest Damage with high risk → pesticide."""
    action = get_action("Pest Damage", 75, 0.4)
    assert "pesticide" in action.lower()

def test_action_bacterial():
    """Bacterial with high risk → bactericide."""
    action = get_action("Bacterial", 75, 0.4)
    assert "bactericide" in action.lower()

def test_action_irrigation():
    """Low NDVI + moderate risk → irrigation."""
    action = get_action("Stress", 55, 0.2)
    assert "irrigation" in action.lower()

def test_action_inspection():
    """WARNING level, no specific disease action → schedule inspection."""
    action = get_action("Stress", 70, 0.6)
    assert "inspection" in action.lower()

def test_action_clear():
    """Low risk → no action required."""
    action = get_action("Healthy", 15, 0.6)
    assert action == "No action required"

def test_action_fungal_low_risk():
    """Fungal Disease but risk ≤ 60 → NOT fungicide (falls to lower rule)."""
    action = get_action("Fungal Disease", 50, 0.6)
    assert "fungicide" not in action.lower()

def test_action_irrigation_priority():
    """NDVI < 0.35 and risk > 50 only applies when no class-specific rule fires."""
    # Stress class doesn't trigger top-3 rules → irrigation rule fires
    action = get_action("Stress", 55, 0.3)
    assert "irrigation" in action.lower()


# ---------------------------------------------------------------------------
# build_alerts + summarise tests
# ---------------------------------------------------------------------------

def test_build_alerts_count():
    mock = {str(i): {"risk_score": i, "cnn_class": "Healthy",
                     "confidence": 0.9, "lstm_ndvi": 0.5}
            for i in [10, 50, 70, 90]}
    alerts = build_alerts(mock)
    assert len(alerts) == 4

def test_build_alerts_levels():
    mock = {
        "a": {"risk_score": 90, "cnn_class": "Fungal Disease",
              "confidence": 0.9, "lstm_ndvi": 0.4},
        "b": {"risk_score": 20, "cnn_class": "Healthy",
              "confidence": 0.9, "lstm_ndvi": 0.6},
    }
    alerts = build_alerts(mock)
    levels = {a["zone_id"]: a["alert_level"] for a in alerts}
    assert levels["a"] == "CRITICAL"
    assert levels["b"] == "CLEAR"

def test_summarise():
    alerts = [
        {"alert_level": "CRITICAL"},
        {"alert_level": "CRITICAL"},
        {"alert_level": "WARNING"},
        {"alert_level": "CLEAR"},
    ]
    s = summarise(alerts)
    assert s["CRITICAL"] == 2
    assert s["WARNING"]  == 1
    assert s["CLEAR"]    == 1
    assert s["WATCH"]    == 0
