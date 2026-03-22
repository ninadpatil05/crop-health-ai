"""
src/alerts/report_generator.py
================================
Generate an A4 PDF field-health report combining risk scores, CNN
classification, LSTM NDVI trends, and sensor data.

Usage
-----
    python -m src.alerts.report_generator
    # or call from Streamlit:
    from src.alerts.report_generator import generate_report
    path = generate_report()
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
RISK_JSON     = PROJECT_ROOT / "outputs" / "maps"    / "risk_scores.json"
SENSOR_CSV    = PROJECT_ROOT / "data"    / "sensor"  / "sensor_data.csv"
ACTIVE_JSON   = PROJECT_ROOT / "outputs" / "alerts"  / "active_alerts.json"
RISK_MAP_PNG  = PROJECT_ROOT / "outputs" / "maps"    / "risk_map.png"
LSTM_PLOT_PNG = PROJECT_ROOT / "outputs" / "metrics" / "lstm_predictions.png"

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
NAVY     = colors.HexColor("#1A3A6B")
LIGHT_BG = colors.HexColor("#EEF2FF")
AMBER    = colors.HexColor("#F59E0B")
RED      = colors.HexColor("#EF4444")
GREEN    = colors.HexColor("#22C55E")
GREY     = colors.grey
WHITE    = colors.white


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _level_color(level: str) -> colors.Color:
    return {"CRITICAL": RED, "WARNING": AMBER,
            "WATCH": AMBER, "CLEAR": GREEN}.get(level, GREY)


def _score_color(score: int) -> colors.Color:
    if score > 80: return RED
    if score > 60: return AMBER
    if score > 30: return AMBER
    return GREEN


def _rule_recommendations(risk_scores: dict, active: list) -> str:
    """Generate plain-English recommendations from risk data."""
    n_crit  = sum(1 for a in active if a.get("alert_level") == "CRITICAL")
    n_warn  = sum(1 for a in active if a.get("alert_level") == "WARNING")
    classes = [v["cnn_class"] for v in risk_scores.values()]
    fungal  = classes.count("Fungal Disease")
    pest    = classes.count("Pest Damage")
    bact    = classes.count("Bacterial")
    stress  = classes.count("Stress")

    lines = []
    if n_crit:
        lines.append(f"• <b>{n_crit} CRITICAL zone(s)</b> require immediate intervention — "
                     "deploy field teams within 24 hours.")
    if n_warn:
        lines.append(f"• <b>{n_warn} WARNING zone(s)</b> require inspection within 72 hours.")
    if fungal:
        lines.append(f"• {fungal} zone(s) show Fungal Disease — apply preventive fungicide "
                     "and ensure adequate canopy airflow.")
    if pest:
        lines.append(f"• {pest} zone(s) show Pest Damage — schedule pesticide spray and "
                     "deploy pheromone traps.")
    if bact:
        lines.append(f"• {bact} zone(s) show Bacterial infection — remove infected plant "
                     "material and apply copper-based bactericide.")
    if stress:
        lines.append(f"• {stress} zone(s) under Stress — review irrigation schedule and "
                     "check soil nutrient levels.")
    if not lines:
        lines.append("• All zones are within acceptable thresholds. Continue routine monitoring.")
    lines.append("• Next full field survey recommended within <b>7 days</b>.")
    return "<br/>".join(lines)


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------

def generate_report(output_dir: str = "outputs/reports") -> str:
    """
    Build and save the A4 PDF field report.

    Parameters
    ----------
    output_dir : Directory to save the PDF (relative to project root or absolute).

    Returns
    -------
    Absolute path to the saved PDF file.
    """
    out_dir  = Path(output_dir) if Path(output_dir).is_absolute() \
               else PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = out_dir / f"field_report_{date_str}.pdf"

    doc   = SimpleDocTemplate(str(out_path), pagesize=A4,
                              leftMargin=2*cm, rightMargin=2*cm,
                              topMargin=2*cm,  bottomMargin=2*cm)
    story = []
    styles= getSampleStyleSheet()

    # -- Custom styles ---------------------------------------------------
    header_style = ParagraphStyle(
        "Header", parent=styles["Title"],
        textColor=NAVY, fontSize=20, spaceAfter=4)
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        textColor=GREY, fontSize=10, spaceAfter=2)
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        textColor=NAVY, fontSize=13, spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9, leading=14)
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        textColor=GREY, fontSize=8, alignment=1)

    # ── 1. Header ────────────────────────────────────────────────────────
    story.append(Paragraph("🌾 Crop Health Field Report", header_style))
    story.append(Paragraph(
        f"Date: {datetime.now().strftime('%B %d, %Y')} &nbsp;|&nbsp; "
        f"Location: Shirpur, MH", sub_style))
    story.append(Spacer(1, 20))

    # ── 2. Load data ─────────────────────────────────────────────────────
    risk_scores: dict = {}
    if RISK_JSON.exists():
        with open(RISK_JSON) as f:
            risk_scores = json.load(f)

    active_alerts: list = []
    if ACTIVE_JSON.exists():
        with open(ACTIVE_JSON) as f:
            active_alerts = json.load(f)

    sensor_row: dict = {}
    if SENSOR_CSV.exists():
        df = pd.read_csv(SENSOR_CSV)
        sensor_row = df.iloc[-1].to_dict() if not df.empty else {}

    # ── 3. Executive summary table ────────────────────────────────────────
    story.append(Paragraph("Executive Summary", section_style))
    scores     = [v["risk_score"] for v in risk_scores.values()] if risk_scores else [0]
    mean_ndvi  = sum(v.get("lstm_ndvi", 0) for v in risk_scores.values()) / max(len(risk_scores), 1)
    high_zones = sum(1 for s in scores if s > 60)
    n_active   = len(active_alerts)

    exec_data = [
        ["Metric", "Value"],
        ["Mean Forecast NDVI",    f"{mean_ndvi:.3f}"],
        ["Mean Risk Score",       f"{sum(scores)/len(scores):.1f} / 100"],
        ["High Risk Zones (>60)", str(high_zones)],
        ["Active Alerts",         str(n_active)],
        ["Total Zones Monitored", str(len(risk_scores))],
    ]
    exec_table = Table(exec_data, colWidths=[10*cm, 6*cm])
    exec_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID",         (0, 0), (-1, -1), 0.5, GREY),
        ("ALIGN",        (1, 0), (1, -1),  "CENTER"),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 12))

    # ── 4. Risk heatmap image ─────────────────────────────────────────────
    if RISK_MAP_PNG.exists():
        story.append(Paragraph("Field Risk Heatmap (10×10 Zone Grid)", section_style))
        story.append(RLImage(str(RISK_MAP_PNG), width=400, height=300))
        story.append(Spacer(1, 10))

    # ── 5. Top 10 high-risk zones table ──────────────────────────────────
    story.append(Paragraph("Top 10 High Risk Zones", section_style))
    sorted_zones = sorted(risk_scores.items(),
                          key=lambda x: x[1]["risk_score"], reverse=True)[:10]

    # Match alert level from active_alerts by zone_id
    alert_map = {str(a["zone_id"]): a for a in active_alerts}

    zone_header = ["Zone", "Risk Score", "CNN Class", "Alert Level", "Action"]
    zone_rows   = [zone_header]
    for zid, info in sorted_zones:
        alert = alert_map.get(str(zid), {})
        level  = alert.get("alert_level", "CLEAR")
        action = alert.get("action", "Monitor")
        # Truncate long actions for table
        action_short = action[:45] + "…" if len(action) > 45 else action
        zone_rows.append([
            f"Zone {zid}",
            str(info["risk_score"]),
            info["cnn_class"],
            level,
            action_short,
        ])

    zone_table = Table(zone_rows, colWidths=[2.2*cm, 2.2*cm, 3.5*cm, 2.5*cm, 6.3*cm])
    zone_ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.4, GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN",         (1, 0), (1, -1),  "CENTER"),
        ("ALIGN",         (3, 0), (3, -1),  "CENTER"),
    ])
    # Alternating row colours
    for i in range(1, len(zone_rows)):
        bg = LIGHT_BG if i % 2 == 0 else WHITE
        zone_ts.add("BACKGROUND", (0, i), (-1, i), bg)
    zone_table.setStyle(zone_ts)
    story.append(zone_table)
    story.append(Spacer(1, 12))

    # ── 6. NDVI trend image ───────────────────────────────────────────────
    if LSTM_PLOT_PNG.exists():
        story.append(Paragraph("LSTM NDVI Forecast — Selected Zones", section_style))
        story.append(RLImage(str(LSTM_PLOT_PNG), width=430, height=220))
        story.append(Spacer(1, 10))

    # ── 7. Sensor summary table ───────────────────────────────────────────
    story.append(Paragraph("Latest Sensor Readings", section_style))
    sensor_data = [
        ["Parameter", "Value", "Status"],
        ["Soil Moisture (m³/m³)", f"{sensor_row.get('soil_moisture', 'N/A'):.3f}" if sensor_row else "N/A",
         "⚠ Low" if float(sensor_row.get("soil_moisture", 0.3)) < 0.20 else "✓ OK"],
        ["Temperature (°C)",      f"{sensor_row.get('temperature', 'N/A'):.1f}" if sensor_row else "N/A",
         "⚠ High" if float(sensor_row.get("temperature", 25)) > 38 else "✓ OK"],
        ["Humidity (%)",          f"{sensor_row.get('humidity', 'N/A'):.1f}" if sensor_row else "N/A",
         "⚠ High" if float(sensor_row.get("humidity", 50)) > 85 else "✓ OK"],
        ["Precipitation (mm)",    f"{sensor_row.get('precipitation', 'N/A'):.2f}" if sensor_row else "N/A",
         "✓ OK"],
        ["Evapotranspiration (mm)",f"{sensor_row.get('evapotranspiration', 'N/A'):.2f}" if sensor_row else "N/A",
         "✓ OK"],
    ]
    sensor_table = Table(sensor_data, colWidths=[6.5*cm, 4*cm, 4*cm])
    sensor_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID",          (0, 0), (-1, -1), 0.5, GREY),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(sensor_table)
    story.append(Spacer(1, 14))

    # ── 8. Recommendations ────────────────────────────────────────────────
    story.append(Paragraph("Recommendations", section_style))
    reco_text = _rule_recommendations(risk_scores, active_alerts)
    story.append(Paragraph(reco_text, body_style))
    story.append(Spacer(1, 20))

    # ── 9. Footer ─────────────────────────────────────────────────────────
    story.append(Paragraph(
        "Generated by <b>CropHealthAI</b> &nbsp;|&nbsp; "
        "MathWorks India PS-25099 &nbsp;|&nbsp; "
        f"Report date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        footer_style,
    ))

    # ── Build ─────────────────────────────────────────────────────────────
    doc.build(story)
    print(f"Report saved: {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_report()
