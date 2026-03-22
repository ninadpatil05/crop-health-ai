"""
CropHealth AI — SaaS Level Dashboard
Premium dark SaaS design: indigo system, glassmorphism, micro-interactions
"""

import os, json, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

import torch
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="CropHealth AI", layout="wide",
                   page_icon="🌾", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{
  --i5:#6366f1;--i4:#818cf8;--i6:#4f46e5;
  --s8:#1e293b;--s85:#172033;--s9:#0f172a;--s95:#080d18;
  --em:#10b981;--am:#f59e0b;--ro:#f43f5e;--cy:#06b6d4;--vi:#8b5cf6;
  --tp:#f1f5f9;--ts:#94a3b8;--tm:#475569;
  --bd:rgba(148,163,184,0.08);--bdh:rgba(99,102,241,0.3);
  --rm:12px;--rl:16px;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
  background:var(--s95)!important;font-family:'Inter',sans-serif!important;color:var(--tp)!important;}
[data-testid="stAppViewContainer"]::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(99,102,241,0.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(99,102,241,0.03) 1px,transparent 1px);
  background-size:40px 40px;pointer-events:none;z-index:0;}
#MainMenu,footer,header,.stDeployButton,[data-testid="stToolbar"]{display:none!important;}
[data-testid="stSidebar"]{background:var(--s9)!important;border-right:1px solid var(--bd)!important;width:258px!important;}
[data-testid="stSidebar"]>div:first-child{padding:0!important;}
::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--i6);border-radius:2px;}
[data-testid="stMainBlockContainer"]{padding:24px 32px!important;max-width:1400px!important;}
.logo{padding:22px 18px 18px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:10px;}
.logo-icon{width:34px;height:34px;background:linear-gradient(135deg,var(--i6),var(--vi));
  border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;}
.logo-name{font-family:'Plus Jakarta Sans',sans-serif;font-size:0.97rem;font-weight:700;color:var(--tp);}
.logo-sub{font-size:0.6rem;color:var(--i4);font-weight:500;letter-spacing:0.04em;}
.nav-section{font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
  color:var(--tm);font-weight:600;padding:18px 18px 6px;}
.stRadio>div{gap:2px!important;padding:0 8px!important;}
.stRadio label{background:transparent!important;border:1px solid transparent!important;
  border-radius:8px!important;padding:9px 12px!important;transition:all 0.15s!important;
  color:var(--ts)!important;font-size:0.855rem!important;font-weight:500!important;}
.stRadio label:hover{background:rgba(99,102,241,0.08)!important;color:var(--tp)!important;border-color:var(--bdh)!important;}
.sb-meta{padding:16px 18px;border-top:1px solid var(--bd);}
.sb-meta-title{font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--tm);font-weight:600;margin-bottom:8px;}
.sb-row{display:flex;justify-content:space-between;padding:4px 0;font-size:0.77rem;}
.sb-lbl{color:var(--tm);}.sb-val{color:var(--tp);font-weight:600;}
.status-dot{width:6px;height:6px;background:var(--em);border-radius:50%;
  display:inline-block;animation:bl 2.5s infinite;margin-right:5px;}
@keyframes bl{0%,100%{opacity:1;}50%{opacity:0.25;}}
.topbar{display:flex;align-items:center;justify-content:space-between;
  padding:0 0 22px;border-bottom:1px solid var(--bd);margin-bottom:24px;position:relative;z-index:1;}
.topbar h1{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.45rem;font-weight:700;
  color:var(--tp);margin:0 0 3px;letter-spacing:-0.02em;}
.topbar p{font-size:0.8rem;color:var(--ts);margin:0;}
.chip{background:var(--s8);border:1px solid var(--bd);border-radius:20px;
  padding:5px 12px;font-size:0.77rem;color:var(--ts);display:inline-flex;align-items:center;gap:5px;}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px;z-index:1;position:relative;}
.kpi{background:var(--s8);border:1px solid var(--bd);border-radius:var(--rl);
  padding:18px 20px;position:relative;overflow:hidden;transition:all 0.2s;animation:fu 0.45s ease both;}
.kpi:hover{border-color:var(--bdh);transform:translateY(-2px);box-shadow:0 8px 28px rgba(99,102,241,0.18);}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--i5),transparent);opacity:0;transition:0.2s;}
.kpi:hover::before{opacity:1;}
.kpi-ico{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;
  justify-content:center;font-size:14px;margin-bottom:12px;}
.ico-i{background:rgba(99,102,241,0.15);}
.ico-e{background:rgba(16,185,129,0.15);}
.ico-a{background:rgba(245,158,11,0.15);}
.ico-r{background:rgba(244,63,94,0.15);}
.kpi-v{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.85rem;font-weight:700;
  color:var(--tp);line-height:1;letter-spacing:-0.03em;margin-bottom:3px;}
.kpi-l{font-size:0.77rem;color:var(--ts);font-weight:500;}
.kpi-t{font-size:0.7rem;margin-top:8px;display:inline-flex;align-items:center;
  gap:4px;padding:2px 7px;border-radius:20px;font-weight:500;}
.t-g{background:rgba(16,185,129,0.1);color:#10b981;}
.t-r{background:rgba(244,63,94,0.1);color:#f43f5e;}
.t-b{background:rgba(99,102,241,0.1);color:#818cf8;}
@keyframes fu{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
.kpi:nth-child(1){animation-delay:.05s;}.kpi:nth-child(2){animation-delay:.10s;}
.kpi:nth-child(3){animation-delay:.15s;}.kpi:nth-child(4){animation-delay:.20s;}
.card{background:var(--s8);border:1px solid var(--bd);border-radius:var(--rl);
  padding:20px 22px;position:relative;z-index:1;margin-bottom:14px;animation:fu .5s ease both;}
.card:hover{border-color:rgba(148,163,184,0.13);}
.ch{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;}
.ct{font-family:'Plus Jakarta Sans',sans-serif;font-size:0.9rem;font-weight:600;
  color:var(--tp);display:flex;align-items:center;gap:7px;}
.cb{background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);
  color:var(--i4);font-size:0.69rem;font-weight:600;padding:2px 9px;border-radius:20px;}
.sbadge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:20px;
  font-size:0.69rem;font-weight:600;letter-spacing:0.02em;white-space:nowrap;}
.bc{background:rgba(244,63,94,0.1);color:#f43f5e;border:1px solid rgba(244,63,94,0.2);}
.bw{background:rgba(245,158,11,0.1);color:#f59e0b;border:1px solid rgba(245,158,11,0.2);}
.bwt{background:rgba(6,182,212,0.1);color:#06b6d4;border:1px solid rgba(6,182,212,0.2);}
.bl{background:rgba(16,185,129,0.1);color:#10b981;border:1px solid rgba(16,185,129,0.2);}
.al{display:flex;align-items:flex-start;gap:10px;padding:12px 14px;
  border-radius:10px;margin-bottom:7px;border-left:3px solid;transition:all 0.15s;animation:si .3s ease both;}
.al:hover{transform:translateX(3px);}
.al-c{background:rgba(244,63,94,0.06);border-color:#f43f5e;}
.al-w{background:rgba(245,158,11,0.06);border-color:#f59e0b;}
.al-wt{background:rgba(6,182,212,0.06);border-color:#06b6d4;}
.al-ico{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;
  justify-content:center;font-size:13px;flex-shrink:0;}
.aic{background:rgba(244,63,94,0.15);}
.aiw{background:rgba(245,158,11,0.15);}
.aiwt{background:rgba(6,182,212,0.12);}
.az{font-weight:600;font-size:0.83rem;color:var(--tp);margin-bottom:1px;}
.aa{font-size:0.75rem;color:var(--ts);line-height:1.4;}
@keyframes si{from{opacity:0;transform:translateX(-6px);}to{opacity:1;transform:translateX(0);}}
.zs-num{font-family:'Plus Jakarta Sans',sans-serif;font-size:2.8rem;font-weight:800;
  letter-spacing:-0.04em;line-height:1;text-align:center;}
.zs-lbl{font-size:0.7rem;color:var(--tm);text-align:center;text-transform:uppercase;
  letter-spacing:0.08em;margin-top:3px;}
.zd{display:flex;justify-content:space-between;padding:8px 0;
  border-bottom:1px solid var(--bd);font-size:0.79rem;}
.zd:last-child{border-bottom:none;}
.zdl{color:var(--tm);font-weight:500;}.zdr{color:var(--tp);font-weight:600;}
.pt{height:4px;background:rgba(148,163,184,0.08);border-radius:2px;overflow:hidden;margin:4px 0 2px;}
.pf{height:100%;border-radius:2px;transition:width 1s cubic-bezier(0.4,0,0.2,1);}
.pr{display:flex;justify-content:space-between;font-size:0.74rem;color:var(--ts);margin-bottom:4px;}
.stButton>button{background:var(--i6)!important;color:white!important;
  border:none!important;border-radius:8px!important;font-weight:600!important;
  font-size:0.82rem!important;padding:0.5rem 1.2rem!important;transition:all 0.2s!important;}
.stButton>button:hover{background:var(--i5)!important;transform:translateY(-1px)!important;
  box-shadow:0 4px 14px rgba(99,102,241,0.35)!important;}
.stDownloadButton>button{background:rgba(99,102,241,0.1)!important;color:var(--i4)!important;
  border:1px solid rgba(99,102,241,0.22)!important;border-radius:8px!important;
  font-weight:600!important;font-size:0.82rem!important;transition:all 0.2s!important;}
.stDownloadButton>button:hover{background:rgba(99,102,241,0.18)!important;border-color:var(--i5)!important;}
.stSelectbox>div>div{background:var(--s85)!important;border:1px solid var(--bd)!important;
  border-radius:8px!important;color:var(--tp)!important;font-size:0.84rem!important;}
.metric-perf{background:var(--s8);border:1px solid var(--bd);border-radius:var(--rl);
  padding:16px 18px;text-align:center;transition:all 0.2s;}
.metric-perf:hover{border-color:var(--bdh);transform:translateY(-2px);}
.mp-val{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.5rem;font-weight:700;margin-bottom:2px;}
.mp-lbl{font-size:0.75rem;color:var(--ts);margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)

def load_json(path, default=None):
    if default is None: default={}
    try: return json.load(open(path))
    except: return default

risk_scores   = load_json("outputs/maps/risk_scores.json")
active_alerts = load_json("outputs/alerts/active_alerts.json",[])
if isinstance(active_alerts,dict): active_alerts=list(active_alerts.values())
sensor_df  = pd.read_csv("data/sensor/sensor_data.csv") if Path("data/sensor/sensor_data.csv").exists() else None
ts_data    = np.load("data/sentinel2/timeseries.npz") if Path("data/sentinel2/timeseries.npz").exists() else None

scores_list     = [v.get("risk_score",0) for v in risk_scores.values()] if risk_scores else [0]
mean_ndvi       = round(float(np.mean([v.get("lstm_ndvi",0) for v in risk_scores.values()])) if risk_scores else 0,3)
high_risk_count = sum(1 for s in scores_list if s>60)
critical_count  = sum(1 for s in scores_list if s>80)
total_zones     = len(risk_scores)

def pdark(fig,h=340):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,23,42,0.5)",
        font=dict(family="Inter",color="#94a3b8",size=11),height=h,
        margin=dict(t=10,b=36,l=8,r=8),
        xaxis=dict(gridcolor="rgba(148,163,184,0.05)",showline=False,zeroline=False,color="#475569"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.05)",showline=False,zeroline=False,color="#475569"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color="#94a3b8")),
        hovermode="x unified",hoverlabel=dict(bgcolor="#1e293b",font_color="white",bordercolor="#334155"))
    return fig

def bdg(lvl):
    m={"CRITICAL":"bc","WARNING":"bw","WATCH":"bwt","CLEAR":"bl"}
    c=m.get(str(lvl).upper(),"bl")
    return f'<span class="sbadge {c}">● {lvl}</span>'

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='logo'>
      <div class='logo-icon'>🌾</div>
      <div><div class='logo-name'>CropHealth AI</div>
      <div class='logo-sub'>MathWorks · PS-25099</div></div>
    </div>
    <div class='nav-section'>Navigation</div>
    """, unsafe_allow_html=True)
    page = st.radio("", ["📊  Dashboard","📈  Analytics","🚨  Alert Center","🗺️  Map View","📄  Reports"],
                    label_visibility="collapsed")
    now = datetime.now().strftime("%d %b · %H:%M")
    st.markdown(f"""
    <div class='sb-meta'>
      <div class='sb-meta-title'>System</div>
      <div class='sb-row'><span class='sb-lbl'><span class='status-dot'></span>Status</span>
        <span class='sb-val'>Live</span></div>
      <div class='sb-row'><span class='sb-lbl'>Time</span><span class='sb-val'>{now}</span></div>
      <div class='sb-row'><span class='sb-lbl'>Alerts</span>
        <span class='sb-val' style='color:#f59e0b;'>{len(active_alerts)}</span></div>
      <div class='sb-row'><span class='sb-lbl'>Zones</span><span class='sb-val'>{total_zones}</span></div>
    </div>""", unsafe_allow_html=True)
    if sensor_df is not None:
        lat=sensor_df.iloc[-1]
        st.markdown(f"""
        <div class='sb-meta'>
          <div class='sb-meta-title'>Live Sensors · Shirpur</div>
          <div class='sb-row'><span class='sb-lbl'>🌡️ Temp</span><span class='sb-val'>{lat.get('temperature',0):.1f}°C</span></div>
          <div class='sb-row'><span class='sb-lbl'>💧 Soil</span><span class='sb-val'>{lat.get('soil_moisture',0):.3f}</span></div>
          <div class='sb-row'><span class='sb-lbl'>💦 Humidity</span><span class='sb-val'>{lat.get('humidity',0):.0f}%</span></div>
          <div class='sb-row'><span class='sb-lbl'>🌧️ Rain</span><span class='sb-val'>{lat.get('precipitation',0):.1f}mm</span></div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div style='padding:12px 18px;font-size:0.66rem;color:#334155;'>ninadpatil05 · PyTorch · Streamlit · 2026</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown(f"""
    <div class='topbar'>
      <div><h1>Field Dashboard</h1><p>Shirpur, Maharashtra · Sentinel-2 T43QCA · Dec 2025</p></div>
      <div style='display:flex;gap:8px;'>
        <span class='chip'><span class='status-dot'></span>Live</span>
        <span class='chip'>📅 {datetime.now().strftime('%d %b %Y')}</span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='kpi-grid'>
      <div class='kpi'><div class='kpi-ico ico-i'>🌿</div>
        <div class='kpi-v'>{mean_ndvi}</div><div class='kpi-l'>Mean NDVI</div>
        <span class='kpi-t t-b'>{'Healthy' if mean_ndvi>0.4 else 'Stressed'}</span></div>
      <div class='kpi'><div class='kpi-ico ico-a'>⚠️</div>
        <div class='kpi-v'>{high_risk_count}</div><div class='kpi-l'>High Risk Zones</div>
        <span class='kpi-t t-r'>of {total_zones} zones</span></div>
      <div class='kpi'><div class='kpi-ico ico-r'>🚨</div>
        <div class='kpi-v'>{len(active_alerts)}</div><div class='kpi-l'>Active Alerts</div>
        <span class='kpi-t t-r'>{critical_count} critical</span></div>
      <div class='kpi'><div class='kpi-ico ico-e'>🧠</div>
        <div class='kpi-v'>98%</div><div class='kpi-l'>CNN Accuracy</div>
        <span class='kpi-t t-g'>↑ Above target</span></div>
    </div>""", unsafe_allow_html=True)
    c1,c2 = st.columns([1.5,1],gap="large")
    with c1:
        st.markdown("<div class='card'><div class='ch'><div class='ct'>🗺️ Risk Zone Heatmap</div><span class='cb'>100 Zones</span></div>",unsafe_allow_html=True)
        if Path("outputs/maps/risk_map.png").exists():
            st.image("outputs/maps/risk_map.png",use_column_width=True)
        else:
            st.info("Run P4-01 to generate risk map")
        st.markdown("</div>",unsafe_allow_html=True)
        if risk_scores and HAS_PLOTLY:
            counts={"CLEAR":0,"WATCH":0,"WARNING":0,"CRITICAL":0}
            for v in risk_scores.values():
                s=v.get("risk_score",0)
                if s>80: counts["CRITICAL"]+=1
                elif s>60: counts["WARNING"]+=1
                elif s>30: counts["WATCH"]+=1
                else: counts["CLEAR"]+=1
            fig=go.Figure(go.Bar(x=list(counts.values()),y=list(counts.keys()),orientation='h',
                marker_color=["#10b981","#06b6d4","#f59e0b","#f43f5e"],
                text=list(counts.values()),textposition="outside",textfont=dict(color="#94a3b8",size=11)))
            pdark(fig,200); fig.update_layout(margin=dict(t=5,b=5,l=5,r=40))
            st.markdown("<div class='card'><div class='ch'><div class='ct'>📊 Risk Distribution</div></div>",unsafe_allow_html=True)
            st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='ch'><div class='ct'>🔍 Zone Inspector</div></div>",unsafe_allow_html=True)
        zone=st.selectbox("Zone",range(100),key="d_zone")
        zk=str(zone)
        if zk in risk_scores:
            z=risk_scores[zk]; rs=z.get("risk_score",0)
            rc,lv=("#f43f5e","CRITICAL") if rs>80 else ("#f59e0b","WARNING") if rs>60 else ("#06b6d4","WATCH") if rs>30 else ("#10b981","CLEAR")
            st.markdown(f"""
            <div style='text-align:center;padding:14px 0 8px;'>
              <div class='zs-num' style='color:{rc};'>{rs}</div>
              <div class='zs-lbl'>Risk Score / 100</div>
              <div style='margin:8px auto;'><div class='pt'><div class='pf' style='width:{rs}%;background:linear-gradient(90deg,{rc}88,{rc});'></div></div></div>
              <div style='margin-top:6px;'>{bdg(lv)}</div>
            </div>
            <div style='margin-top:10px;'>
              <div class='zd'><span class='zdl'>CNN Class</span><span class='zdr'>{z.get('cnn_class','—')}</span></div>
              <div class='zd'><span class='zdl'>Confidence</span><span class='zdr'>{z.get('confidence',0)*100:.0f}%</span></div>
              <div class='zd'><span class='zdl'>LSTM NDVI</span><span class='zdr'>{z.get('lstm_ndvi',0):.3f}</span></div>
              <div class='zd'><span class='zdl'>Soil Moisture</span><span class='zdr'>{z.get('soil_moisture',0):.3f}</span></div>
              <div class='zd'><span class='zdl'>Humidity</span><span class='zdr'>{z.get('humidity',0):.0f}%</span></div>
            </div>""",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='ch'><div class='ct'>🚨 Recent Alerts</div></div>",unsafe_allow_html=True)
        shown=[a for a in active_alerts if a.get("alert_level","").upper() in ("CRITICAL","WARNING")][:5]
        if shown:
            for a in shown:
                lv=a.get("alert_level","WATCH").upper(); zid=a.get("zone_id","?")
                act=a.get("recommended_action","Inspect")[:52]+"…"
                ic={"CRITICAL":"🔴","WARNING":"🟡","WATCH":"🔵"}.get(lv,"⚪")
                ac={"CRITICAL":"al-c","WARNING":"al-w","WATCH":"al-wt"}.get(lv,"al-wt")
                ic2={"CRITICAL":"aic","WARNING":"aiw","WATCH":"aiwt"}.get(lv,"aiwt")
                st.markdown(f"""<div class='al {ac}'>
                  <div class='al-ico {ic2}'>{ic}</div>
                  <div><div class='az'>Zone {zid} {bdg(lv)}</div>
                  <div class='aa'>{act}</div></div></div>""",unsafe_allow_html=True)
        else:
            st.success("✅ No critical alerts")
        st.markdown("</div>",unsafe_allow_html=True)

elif "Analytics" in page:
    st.markdown("""<div class='topbar'><div><h1>NDVI Analytics</h1>
      <p>Temporal analysis · LSTM 2-week forecast</p></div></div>""",unsafe_allow_html=True)
    zone=st.selectbox("Select Zone",range(100),key="a_zone")
    if ts_data is not None and HAS_PLOTLY:
        zn=ts_data.get("zone_ndvi",None)
        if zn is not None and zone<len(zn):
            s=zn[zone]; n=len(s); wk=list(range(1,n+1))
            fc=None
            try:
                from src.training.lstm_model import NDVIForecaster
                fp="models/fusion/lstm_fused.pt"
                if Path(fp).exists():
                    ck=torch.load(fp,map_location="cpu",weights_only=False)
                    md=NDVIForecaster(input_size=6); md.load_state_dict(ck["model_state_dict"]); md.eval()
                    sq=torch.FloatTensor(np.concatenate([s[-8:].reshape(1,8,1),np.zeros((1,8,5))],axis=2))
                    with torch.no_grad(): fc=md(sq).numpy()[0]
            except: pass
            curr=s[-1]; chg=s[-1]-s[-2] if n>1 else 0
            cols=st.columns(4)
            for co,lb,vl,cl in [(cols[0],"Current NDVI",f"{curr:.3f}","#6366f1"),
                (cols[1],"Week Change",f"{'+' if chg>=0 else ''}{chg:.3f}","#10b981" if chg>=0 else "#f43f5e"),
                (cols[2],"52-wk Mean",f"{s.mean():.3f}","#06b6d4"),
                (cols[3],"Forecast W1",f"{fc[0]:.3f}" if fc is not None else "—","#f59e0b")]:
                co.markdown(f"<div class='kpi' style='margin-bottom:14px;'><div class='kpi-v' style='color:{cl};font-size:1.5rem;'>{vl}</div><div class='kpi-l'>{lb}</div></div>",unsafe_allow_html=True)
            fig=go.Figure()
            fig.add_hrect(y0=0.4,y1=0.8,fillcolor="rgba(16,185,129,0.04)",line_width=0,
                annotation_text="Healthy",annotation_font=dict(color="#10b981",size=10),annotation_position="top left")
            fig.add_trace(go.Scatter(x=wk,y=s,mode="none",fill="tozeroy",fillcolor="rgba(99,102,241,0.06)",showlegend=False))
            fig.add_trace(go.Scatter(x=wk,y=s,mode="lines+markers",name="NDVI History",
                line=dict(color="#6366f1",width=2),marker=dict(size=4,color="#818cf8",line=dict(color="#0f172a",width=1))))
            if fc is not None:
                fx=[n,n+1,n+2]; fy=[s[-1],fc[0],fc[1]]
                fig.add_trace(go.Scatter(x=fx,y=fy,mode="lines+markers",name="LSTM Forecast",
                    line=dict(color="#f59e0b",width=2,dash="dash"),marker=dict(size=7,color="#f59e0b",symbol="diamond")))
                fig.add_trace(go.Scatter(x=fx+fx[::-1],y=[y+0.03 for y in fy]+[y-0.03 for y in fy[::-1]],
                    fill="toself",fillcolor="rgba(245,158,11,0.06)",line=dict(width=0),name="±0.03",showlegend=True))
            if sensor_df is not None and "soil_moisture" in sensor_df.columns:
                sv=sensor_df["soil_moisture"].tail(n).values
                fig.add_trace(go.Scatter(x=list(range(1,len(sv)+1)),y=sv,mode="lines",name="Soil Moisture",
                    line=dict(color="#06b6d4",width=1.5,dash="dot"),yaxis="y2"))
                fig.update_layout(yaxis2=dict(overlaying="y",side="right",showgrid=False,color="#06b6d4"))
            pdark(fig,380); fig.update_xaxes(title="Week"); fig.update_yaxes(title="NDVI",range=[-0.05,1.05])
            st.markdown(f"<div class='card'><div class='ch'><div class='ct'>📈 Zone {zone} — NDVI Trend & Forecast</div><span class='cb'>LSTM + Sensor</span></div>",unsafe_allow_html=True)
            st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
    if sensor_df is not None and HAS_PLOTLY:
        fig2=go.Figure()
        dc="date" if "date" in sensor_df.columns else list(range(len(sensor_df)))
        for co,cl in [("temperature","#f43f5e"),("humidity","#06b6d4"),("soil_moisture","#10b981"),("precipitation","#8b5cf6")]:
            if co in sensor_df.columns:
                fig2.add_trace(go.Scatter(x=sensor_df[dc] if isinstance(dc,str) else dc,y=sensor_df[co],mode="lines",
                    name=co.replace("_"," ").title(),line=dict(color=cl,width=1.8)))
        pdark(fig2,260)
        st.markdown("<div class='card'><div class='ch'><div class='ct'>🌡️ 90-Day Environmental Sensors</div></div>",unsafe_allow_html=True)
        st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)

elif "Alert" in page:
    st.markdown("""<div class='topbar'><div><h1>Alert Center</h1>
      <p>CNN + LSTM + Sensor triggered alerts</p></div></div>""",unsafe_allow_html=True)
    c1,c2=st.columns([1.4,1],gap="large")
    with c1:
        hr={k:v for k,v in risk_scores.items() if v.get("risk_score",0)>60}
        rows=[{"Zone":f"Zone {zi}","Score":v.get("risk_score",0),"Class":v.get("cnn_class","—"),
               "Conf":f"{v.get('confidence',0)*100:.0f}%","NDVI":f"{v.get('lstm_ndvi',0):.3f}",
               "Level":"CRITICAL" if v.get("risk_score",0)>80 else "WARNING"}
              for zi,v in sorted(hr.items(),key=lambda x:-x[1].get("risk_score",0))]
        st.markdown("<div class='card'><div class='ch'><div class='ct'>⚠️ High Risk Zones</div><span class='cb'>score > 60</span></div>",unsafe_allow_html=True)
        if rows:
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,
                column_config={"Score":st.column_config.ProgressColumn("Score",min_value=0,max_value=100,format="%d")})
        else:
            st.success("✅ All zones clear")
        st.markdown("</div>",unsafe_allow_html=True)
        if risk_scores and HAS_PLOTLY:
            cc={}
            for v in risk_scores.values():
                c=v.get("cnn_class","Unknown"); cc[c]=cc.get(c,0)+1
            fig=go.Figure(go.Bar(x=list(cc.keys()),y=list(cc.values()),
                marker_color=["#6366f1","#f43f5e","#06b6d4","#f59e0b","#8b5cf6"][:len(cc)],
                text=list(cc.values()),textposition="outside",textfont=dict(color="#94a3b8")))
            pdark(fig,220)
            st.markdown("<div class='card'><div class='ch'><div class='ct'>🧬 Disease Distribution</div></div>",unsafe_allow_html=True)
            st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='ch'><div class='ct'>🚨 Active Alerts</div></div>",unsafe_allow_html=True)
        if active_alerts:
            for a in sorted(active_alerts,key=lambda x:x.get("risk_score",0),reverse=True)[:12]:
                lv=a.get("alert_level","WATCH").upper(); zid=a.get("zone_id","?")
                act=a.get("recommended_action","Inspect"); ic={"CRITICAL":"🔴","WARNING":"🟡","WATCH":"🔵"}.get(lv,"⚪")
                ac={"CRITICAL":"al-c","WARNING":"al-w","WATCH":"al-wt"}.get(lv,"al-wt")
                ic2={"CRITICAL":"aic","WARNING":"aiw","WATCH":"aiwt"}.get(lv,"aiwt")
                st.markdown(f"""<div class='al {ac}'>
                  <div class='al-ico {ic2}'>{ic}</div>
                  <div style='flex:1;min-width:0;'>
                    <div class='az'>Zone {zid} {bdg(lv)}<span style='font-size:0.7rem;color:#475569;margin-left:5px;'>Risk:{a.get('risk_score',0)}</span></div>
                    <div class='aa'>{a.get('cnn_class','—')}</div>
                    <div class='aa' style='color:#64748b;margin-top:2px;'>💡 {act[:55]}…</div>
                  </div></div>""",unsafe_allow_html=True)
        else:
            st.success("✅ No active alerts")
        st.markdown("</div>",unsafe_allow_html=True)

elif "Map" in page:
    st.markdown("""<div class='topbar'><div><h1>Map View</h1>
      <p>Interactive field map · Shirpur, Maharashtra</p></div></div>""",unsafe_allow_html=True)
    c1,c2=st.columns([1.6,1],gap="large")
    with c1:
        st.markdown("<div class='card'><div class='ch'><div class='ct'>🗺️ Interactive Risk Map</div><span class='cb'>Folium</span></div>",unsafe_allow_html=True)
        hp=Path("outputs/maps/risk_map.html")
        if hp.exists():
            st.components.v1.html(open(str(hp)).read(),height=420)
        elif Path("outputs/maps/risk_map.png").exists():
            st.image("outputs/maps/risk_map.png",use_column_width=True)
        else:
            st.info("Run P4-01")
        st.markdown("</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='ch'><div class='ct'>📊 Zone Breakdown</div></div>",unsafe_allow_html=True)
        if risk_scores and HAS_PLOTLY:
            cnts={"CLEAR":0,"WATCH":0,"WARNING":0,"CRITICAL":0}
            for v in risk_scores.values():
                s=v.get("risk_score",0)
                if s>80: cnts["CRITICAL"]+=1
                elif s>60: cnts["WARNING"]+=1
                elif s>30: cnts["WATCH"]+=1
                else: cnts["CLEAR"]+=1
            fig=go.Figure(go.Pie(labels=list(cnts.keys()),values=list(cnts.values()),hole=0.58,
                marker_colors=["#10b981","#06b6d4","#f59e0b","#f43f5e"],textfont=dict(color="white",size=11)))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=230,margin=dict(t=0,b=0,l=0,r=0),
                legend=dict(font=dict(color="#94a3b8",size=10),bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(text=f"<b>{total_zones}</b><br>Zones",x=0.5,y=0.5,showarrow=False,
                    font=dict(size=13,color="white"))])
            st.plotly_chart(fig,use_container_width=True)
        for lv,cnt,color in [("Critical",critical_count,"#f43f5e"),
            ("Warning",high_risk_count-critical_count,"#f59e0b"),
            ("Watch",sum(1 for s in scores_list if 30<s<=60),"#06b6d4"),
            ("Clear",sum(1 for s in scores_list if s<=30),"#10b981")]:
            pct=int(cnt/max(total_zones,1)*100)
            st.markdown(f"""<div style='margin-bottom:10px;'>
              <div class='pr'><span style='color:{color};font-weight:600;'>{lv}</span><span>{cnt} ({pct}%)</span></div>
              <div class='pt'><div class='pf' style='width:{pct}%;background:{color};'></div></div>
            </div>""",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

elif "Reports" in page:
    st.markdown("""<div class='topbar'><div><h1>Reports & Export</h1>
      <p>Download field health reports and zone data</p></div></div>""",unsafe_allow_html=True)
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown("""<div class='card'><div class='ch'><div class='ct'>📄 PDF Field Report</div></div>
          <p style='font-size:0.81rem;color:#64748b;line-height:1.7;margin-bottom:16px;'>
          Risk map · NDVI trends · Alert table · Sensor summary · Recommendations</p>""",unsafe_allow_html=True)
        if st.button("Generate PDF Report",use_container_width=True):
            with st.spinner("Building…"):
                try:
                    sys.path.insert(0,".")
                    from src.alerts.report_generator import generate_report
                    path=generate_report()
                    with open(path,"rb") as f:
                        st.download_button("📥 Download PDF",f,
                            file_name=f"field_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",use_container_width=True)
                    st.success(f"✅ {path}")
                except Exception as e:
                    st.error(str(e))
        st.markdown("</div>",unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card'><div class='ch'><div class='ct'>📊 CSV Zone Export</div></div>
          <p style='font-size:0.81rem;color:#64748b;line-height:1.7;margin-bottom:16px;'>
          All 100 zones: NDVI · NDRE · EVI · CNN predictions · Risk scores · Sensor readings</p>""",unsafe_allow_html=True)
        if st.button("Generate CSV Export",use_container_width=True):
            with st.spinner("Aggregating…"):
                try:
                    from src.inference.csv_exporter import export_csv
                    df_ex=export_csv()
                    st.download_button("📥 Download CSV",df_ex.to_csv(index=False).encode(),
                        file_name=f"field_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",use_container_width=True)
                    st.success(f"✅ {len(df_ex)} zones exported")
                    st.dataframe(df_ex.head(4),use_container_width=True,hide_index=True)
                except Exception as e:
                    st.error(str(e))
        st.markdown("</div>",unsafe_allow_html=True)
    st.markdown("<div class='card'><div class='ch'><div class='ct'>🏆 Model Performance</div><span class='cb'>All targets exceeded</span></div>",unsafe_allow_html=True)
    pc=st.columns(4)
    for co,(lb,vl,su,cl,pt) in zip(pc,[
        ("CNN Accuracy","98.02%","16,292 test images","#6366f1",98),
        ("LSTM RMSE","0.013","Target < 0.05","#10b981",97),
        ("R² Score","0.989","Target > 0.85","#06b6d4",99),
        ("Avg F1","0.97","5-class avg","#f59e0b",97)]):
        co.markdown(f"""<div class='metric-perf'>
          <div class='mp-val' style='color:{cl};'>{vl}</div>
          <div class='mp-lbl'>{lb}</div>
          <div class='pt'><div class='pf' style='width:{pt}%;background:{cl};'></div></div>
          <div style='font-size:0.67rem;color:#475569;margin-top:5px;'>{su}</div>
        </div>""",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)
    st.markdown("""<div style='text-align:center;padding:20px 0 6px;border-top:1px solid rgba(148,163,184,0.06);margin-top:6px;'>
      <span style='font-size:0.73rem;color:#334155;'>
        CropHealth AI · MathWorks India PS-25099 · ninadpatil05 · PyTorch · Streamlit · Sentinel-2
      </span></div>""",unsafe_allow_html=True)
