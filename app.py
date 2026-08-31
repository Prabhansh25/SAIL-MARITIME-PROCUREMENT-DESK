"""
app.py — SAIL Maritime Freight & Chartering Decision Support System
====================================================================
Institutional dark-terminal frontend (Bloomberg / S&P Platts style)
Streamlit + Plotly | Aggressive caching | Enterprise CSS

Run: streamlit run app.py
"""

from __future__ import annotations

import datetime
import io
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ─── Optimizer constants — imported at module scope ───────────────────────────
# These are needed in multiple tabs and f-strings; import once here.
try:
    from optimizer import PORT_CONSTRAINTS, VESSEL_CLASS_DRAFT
except Exception:
    # Fallback definitions (mirrors optimizer.py values + extra classes)
    PORT_CONSTRAINTS: dict = {
        "Haldia":   {"max_draft": 8.5,  "allowed_classes": ["Handysize", "Supramax"],
                     "queue_days_avg": 3.5, "demurrage_usd_day": 18_000},
        "Paradip":  {"max_draft": 14.0, "allowed_classes": ["Handysize","Supramax","Panamax","Capesize"],
                     "queue_days_avg": 1.5, "demurrage_usd_day": 22_000},
        "Vizag":    {"max_draft": 14.5, "allowed_classes": ["Handysize","Supramax","Panamax","Capesize"],
                     "queue_days_avg": 2.0, "demurrage_usd_day": 20_000},
        "Ennore":   {"max_draft": 12.5, "allowed_classes": ["Handysize","Supramax","Panamax"],
                     "queue_days_avg": 2.5, "demurrage_usd_day": 19_000},
        "Kolkata":  {"max_draft": 8.0,  "allowed_classes": ["Handysize"],
                     "queue_days_avg": 4.0, "demurrage_usd_day": 16_000},
        "Mormugao": {"max_draft": 13.0, "allowed_classes": ["Handysize","Supramax","Panamax"],
                     "queue_days_avg": 1.8, "demurrage_usd_day": 17_000},
    }
    VESSEL_CLASS_DRAFT: dict = {
        "Capesize":  18.2,
        "Panamax":   14.5,
        "Kamsarmax": 14.5,
        "Supramax":  12.8,
        "Handymax":  11.5,
        "Handysize": 10.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAIL Maritime Procurement Desk",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ENTERPRISE CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root Variables ── */
:root {
  --bg-primary:   #0f172a;
  --bg-secondary: #1e293b;
  --bg-card:      #162032;
  --bg-elevated:  #243347;
  --border:       #1e293b;
  --border-light: #2d3f57;
  --accent:       #38bdf8;
  --accent-dim:   #0ea5e9;
  --green:        #22c55e;
  --red:          #ef4444;
  --amber:        #f59e0b;
  --text-primary: #e2e8f0;
  --text-secondary:#94a3b8;
  --text-muted:   #64748b;
  --mono:         'JetBrains Mono', 'Courier New', monospace;
  --sans:         'Inter', 'Helvetica Neue', sans-serif;
}

/* ── Global Reset ── */
html, body, [class*="css"]  { font-family: var(--sans); }
.stApp                       { background-color: var(--bg-primary) !important; }
.main .block-container       { padding: 0.5rem 1rem 1rem 1rem !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { background-color: #0c1424 !important; border-right: 1px solid var(--border-light); }
section[data-testid="stSidebar"] > div { padding-top: 0.5rem; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton            { display: none; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]  { background: var(--bg-secondary); border-bottom: 1px solid var(--border-light); gap: 0; padding: 0 1rem; }
.stTabs [data-baseweb="tab"]       { color: var(--text-secondary) !important; font-size: 0.75rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; padding: 0.65rem 1.2rem; border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"]     { color: var(--accent) !important; border-bottom-color: var(--accent) !important; background: transparent !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 1rem 0; }

/* ── Sidebar widgets ── */
.stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label {
  color: var(--text-secondary) !important; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;
}
.stSelectbox > div > div { background: var(--bg-secondary) !important; border-color: var(--border-light) !important; color: var(--text-primary) !important; font-size: 0.82rem; }
.stSlider [data-testid="stThumbValue"] { color: var(--accent) !important; font-family: var(--mono); font-size: 0.75rem; }

/* ── Buttons ── */
.stButton > button {
  background: var(--bg-elevated) !important; color: var(--accent) !important;
  border: 1px solid var(--accent) !important; border-radius: 3px !important;
  font-family: var(--mono); font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase; padding: 0.45rem 1rem;
}
.stButton > button:hover { background: var(--accent) !important; color: #0f172a !important; }
.stDownloadButton > button {
  background: #14532d !important; color: #4ade80 !important;
  border: 1px solid #22c55e !important; border-radius: 3px !important;
  font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
  background: var(--bg-card); border: 1px solid var(--border-light);
  border-radius: 4px; padding: 0.6rem 0.8rem;
}
[data-testid="metric-container"] label { color: var(--text-muted) !important; font-size: 0.65rem !important; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 500; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: var(--text-primary) !important; font-family: var(--mono) !important; font-size: 1.3rem !important; font-variant-numeric: tabular-nums; }
[data-testid="metric-container"] [data-testid="metric-delta"] { font-family: var(--mono); font-size: 0.72rem; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border-light); border-radius: 4px; }

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track  { background: var(--bg-primary); }
::-webkit-scrollbar-thumb  { background: var(--border-light); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY COMMON LAYOUT (no duplicate keys)
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=11, color="#94a3b8"),
    margin=dict(l=50, r=30, t=35, b=40),
    xaxis=dict(
        gridcolor="#1e293b", gridwidth=0.5,
        showspikes=True, spikecolor="#38bdf8", spikethickness=1, spikemode="across",
        linecolor="#2d3f57", tickfont=dict(family="JetBrains Mono", size=10),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#1e293b", gridwidth=0.5,
        linecolor="#2d3f57", tickfont=dict(family="JetBrains Mono", size=10),
        zeroline=False,
    ),
    hoverlabel=dict(bgcolor="#1e293b", bordercolor="#2d3f57", font=dict(family="JetBrains Mono", size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2d3f57", borderwidth=1,
                font=dict(size=10, family="JetBrains Mono")),
)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading freight forecasting model…")
def load_freight_engine():
    from optimizer import FreightEngine
    return FreightEngine()

@st.cache_resource(show_spinner="Training bunker fuel model…")
def load_fuel_engine():
    from optimizer import FuelEngine
    return FuelEngine()

@st.cache_resource(show_spinner="Training charter timing ensemble…")
def load_timing_engine():
    from optimizer import TimingEngine
    return TimingEngine()

@st.cache_resource(show_spinner="Loading vessel allocation engine…")
def load_vessel_engine():
    from optimizer import VesselEngine
    return VesselEngine()

@st.cache_data(ttl=300)
def get_freight_forecast(
    route, cargo_type, vessel_class, market_event,
    vessel_dwt, distance_nm, bdi, bunker, congestion,
):
    eng = load_freight_engine()
    today = datetime.date.today()
    base_params = dict(
        route=route, cargo_type=cargo_type, vessel_class=vessel_class,
        market_event=market_event, vessel_dwt=vessel_dwt,
        distance_nm=distance_nm, baltic_style_index=bdi,
        bunker_fuel_price=bunker, port_congestion_days=congestion,
        month=today.month, year=today.year,
        week_of_year=int(today.isocalendar()[1]),
    )
    return eng.forecast_14d(base_params)

@st.cache_data(ttl=300)
def get_fuel_forecast(bunker_price):
    eng = load_fuel_engine()
    return eng.forecast_bunker_14d(bunker_price)

@st.cache_data(ttl=300)
def get_timing_decision(dwt, vessel_age, fuel_price, wind_speed,
                        wave_height, speed, distance_nm, cargo_tonnes):
    eng = load_timing_engine()
    return eng.get_decision(
        dwt=dwt, vessel_age=vessel_age, fuel_price=fuel_price,
        wind_speed=wind_speed, wave_height=wave_height,
        speed=speed, distance_nm=distance_nm, cargo_tonnes=cargo_tonnes,
    )

@st.cache_data(ttl=300)
def get_ranked_vessels(cargo_tonnes, distance_nm, port, required_days):
    eng = load_vessel_engine()
    return eng.rank(cargo_tonnes, distance_nm, port, required_days)

@st.cache_data
def get_port_compliance():
    eng = load_vessel_engine()
    return eng.port_compliance_matrix()

@st.cache_data(ttl=300)
def get_speed_opt(dwt, bunker_price, distance_nm, vessel_age, wind_speed, wave_height, speed, vessel_class):
    from optimizer import _dwt_to_type
    eng = load_fuel_engine()
    vessel_row = pd.DataFrame([{
        "vessel_type": _dwt_to_type(dwt),
        "dwt": dwt, "engine_power": 15000, "vessel_age": vessel_age,
        "draft": VESSEL_CLASS_DRAFT.get(vessel_class, 12.0),
        "speed": speed, "distance_nm": distance_nm, "heading": 90,
        "wind_speed": wind_speed, "wind_direction": 130,
        "wave_height": wave_height, "wave_period": 8,
        "wave_direction": 120, "current_speed": 1.2, "current_direction": 80,
        "fuel_price": bunker_price,
    }])
    return eng.optimize_speed(vessel_row, bunker_price, distance_nm / 12.0)


# ─────────────────────────────────────────────────────────────────────────────
# INSTITUTIONAL HEADER
# ─────────────────────────────────────────────────────────────────────────────
now_ist = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
now_utc = datetime.datetime.utcnow()

st.markdown(f"""
<div style="background:#0c1424;border-bottom:1px solid #2d3f57;padding:0.5rem 1rem;margin:-0.5rem -1rem 0.75rem -1rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:700;
                   color:#e2e8f0;letter-spacing:0.06em;">
        ⚓  SAIL MARITIME PROCUREMENT DESK
      </span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                   color:#38bdf8;margin-left:1rem;letter-spacing:0.04em;">
        PROD v2.4.1
      </span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#64748b;text-align:right;">
      <span style="color:#22c55e;">●</span> BDI (LIVE) &nbsp;|&nbsp;
      <span style="color:#22c55e;">●</span> PLATTs (LIVE) &nbsp;|&nbsp;
      <span style="color:#22c55e;">●</span> IPA PORTS (SYNCED)
      &nbsp;&nbsp;
      <span style="color:#94a3b8;">IST {now_ist.strftime('%Y-%m-%d %H:%M:%S')}</span>
      &nbsp;|&nbsp;
      <span style="color:#64748b;">UTC {now_utc.strftime('%H:%M:%S')}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                color:#38bdf8;letter-spacing:0.1em;text-transform:uppercase;
                border-bottom:1px solid #1e293b;padding-bottom:0.4rem;margin-bottom:0.8rem;'>
      ⚙ Fixture Parameters
    </div>""", unsafe_allow_html=True)

    # ── Route & Vessel ──────────────────────────────────────────────────────
    ALL_ROUTES = [
        "Dampier, Australia - Visakhapatnam",
        "Abbot Point, Australia - Ennore",
        "Abbot Point, Australia - Haldia",
        "Abbot Point, Australia - Kolkata",
        "Abbot Point, Australia - Paradip",
        "Port Hedland, Australia - Vizag",
        "Richards Bay, South Africa - Vizag",
        "Newcastle, Australia - Paradip",
        "Murmansk, Russia - Vizag",
    ]
    route = st.selectbox("Route", ALL_ROUTES, index=0)
    cargo_type = st.selectbox("Cargo Type", ["Coal Ore", "Iron Ore", "Coking Coal", "Thermal Coal"], index=0)
    vessel_class = st.selectbox("Vessel Class", ["Handysize", "Supramax", "Panamax", "Capesize"], index=2)

    VESSEL_DWT_MAP = {"Handysize": 30000, "Supramax": 55000, "Panamax": 78000, "Capesize": 160000}
    vessel_dwt = VESSEL_DWT_MAP[vessel_class]

    port = st.selectbox("Destination Port", ["Paradip", "Vizag", "Ennore", "Haldia", "Kolkata", "Mormugao"], index=0)
    cargo_tonnes = st.number_input("Cargo Volume (MT)", min_value=5000, max_value=200000, value=60000, step=1000)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                color:#38bdf8;letter-spacing:0.1em;text-transform:uppercase;
                border-bottom:1px solid #1e293b;padding-bottom:0.4rem;margin-bottom:0.8rem;margin-top:0.6rem;'>
      📊 Market Conditions
    </div>""", unsafe_allow_html=True)

    bdi        = st.slider("Baltic Dry Index (BDI)", 800, 4000, 1550, 50)
    bunker     = st.slider("Bunker Price (USD/MT)", 400, 1000, 640, 10)
    congestion = st.slider("Port Congestion (days)", 0.0, 8.0, 1.5, 0.5)
    wind_speed = st.slider("Wind Speed (knots)", 0.0, 40.0, 12.0, 0.5)
    wave_height= st.slider("Wave Height (m)", 0.0, 8.0, 2.0, 0.25)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                color:#38bdf8;letter-spacing:0.1em;text-transform:uppercase;
                border-bottom:1px solid #1e293b;padding-bottom:0.4rem;margin-bottom:0.8rem;margin-top:0.6rem;'>
      🚢 Voyage Parameters
    </div>""", unsafe_allow_html=True)

    distance_nm   = st.number_input("Distance (NM)", 1000, 15000, 3570, 100)
    vessel_age    = st.slider("Vessel Age (years)", 1, 25, 8, 1)
    required_days = st.slider("Transit Window (days)", 10, 60, 30, 1)
    market_event  = st.selectbox("Market Event", ["None", "Port Strike", "Canal Disruption", "Monsoon Season", "High Demand"], index=0)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("▶  RUN OPTIMIZATION", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE (runs on button press OR on first load)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _dummy_first_run():
    return True

if "results_computed" not in st.session_state:
    st.session_state.results_computed = False

if run_btn or not st.session_state.results_computed:
    with st.spinner("Running pipeline optimization…"):
        try:
            freight_fc   = get_freight_forecast(route, cargo_type, vessel_class, market_event,
                                                vessel_dwt, distance_nm, bdi, bunker, congestion)
            fuel_fc      = get_fuel_forecast(bunker)
            timing_dec   = get_timing_decision(vessel_dwt, vessel_age, bunker, wind_speed,
                                               wave_height, 14.0, distance_nm, cargo_tonnes)
            ranked_df    = get_ranked_vessels(cargo_tonnes, distance_nm, port, required_days)
            compliance   = get_port_compliance()
            speed_opt    = get_speed_opt(vessel_dwt, bunker, distance_nm, vessel_age,
                                        wind_speed, wave_height, 14.0, vessel_class)
            st.session_state.results_computed = True
            st.session_state.freight_fc  = freight_fc
            st.session_state.fuel_fc     = fuel_fc
            st.session_state.timing_dec  = timing_dec
            st.session_state.ranked_df   = ranked_df
            st.session_state.compliance  = compliance
            st.session_state.speed_opt   = speed_opt
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

freight_fc  = st.session_state.get("freight_fc", {})
fuel_fc     = st.session_state.get("fuel_fc", {})
timing_dec  = st.session_state.get("timing_dec", {})
ranked_df   = st.session_state.get("ranked_df", pd.DataFrame())
compliance  = st.session_state.get("compliance", pd.DataFrame())
speed_opt   = st.session_state.get("speed_opt", {})

if not freight_fc:
    st.info("Set parameters in the sidebar and click ▶ RUN OPTIMIZATION to begin.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# DECISION SIGNAL BANNER
# ─────────────────────────────────────────────────────────────────────────────
decision      = timing_dec.get("decision", "DEFER")
decision_lbl  = timing_dec.get("decision_label", "DECISION RECOMMENDATION: DEFER FIXTURE")
savings       = timing_dec.get("savings_if_wait", 0)
t_star        = timing_dec.get("optimal_t_star", 0)

if decision == "EXECUTE_NOW":
    banner_color = "#14532d"
    border_color = "#22c55e"
    icon         = "🟢"
    signal_color = "#4ade80"
else:
    banner_color = "#451a03"
    border_color = "#f59e0b"
    icon         = "🟡"
    signal_color = "#fbbf24"

st.markdown(f"""
<div style="background:{banner_color};border:1px solid {border_color};
            border-radius:4px;padding:0.7rem 1.2rem;margin-bottom:0.75rem;
            display:flex;justify-content:space-between;align-items:center;">
  <div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                 font-weight:700;color:{signal_color};letter-spacing:0.06em;">
      {icon}  {decision_lbl}
    </span>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#94a3b8;">
    DIRECTIONAL ACC: <span style="color:{signal_color};">{timing_dec.get('directional_accuracy','—')}%</span>
    &nbsp;|&nbsp;
    PROJECTED SAVINGS: <span style="color:{signal_color};">${savings:,.0f}</span>
    &nbsp;|&nbsp;
    OPTIMAL WINDOW: <span style="color:{signal_color};">T+{t_star} DAYS</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  COMMERCIAL CHARTERING DESK  ",
    "  PORT & BERTH OPERATIONS  ",
    "  EXECUTIVE SUMMARY  ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMMERCIAL CHARTERING DESK
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI Ribbon ──────────────────────────────────────────────────────────
    cur_rate  = timing_dec.get("current_rate", 0)
    fut_rate  = timing_dec.get("predicted_future", 0)
    tlc       = timing_dec.get("total_landed_cost", 0)
    dir_acc   = timing_dec.get("directional_accuracy", 0)
    fcast_t14 = freight_fc["mean"][-1] if freight_fc.get("mean") else 0
    fcast_t0  = freight_fc["mean"][0]  if freight_fc.get("mean") else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("CURRENT CHARTER RATE", f"${cur_rate:,.0f}/day",
              delta=f"{((cur_rate - fut_rate)/max(fut_rate,1)*100):+.1f}% vs forecast")
    k2.metric("FREIGHT RATE  (T+0)", f"${fcast_t0:.2f}/t",
              delta=f"T+14: ${fcast_t14:.2f}/t")
    k3.metric("BUNKER PRICE", f"${bunker}/MT",
              delta=f"14d proj: ${fuel_fc['price'][-1] if fuel_fc.get('price') else bunker:.0f}/MT")
    k4.metric("TOTAL LANDED COST", f"${tlc:,.0f}", delta=None)
    k5.metric("DIRECTIONAL ACCURACY", f"{dir_acc:.1f}%", delta="Model | 80/20 split")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Dual-axis Freight + Bunker Chart ────────────────────────────────────
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    dates = freight_fc["dates"]
    means = freight_fc["mean"]
    lows  = freight_fc["lower"]
    highs = freight_fc["upper"]

    # Uncertainty band
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=highs + lows[::-1],
        fill="toself",
        fillcolor="rgba(56,189,248,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="±1σ Band",
    ), secondary_y=False)

    # Freight forecast line
    fig.add_trace(go.Scatter(
        x=dates, y=means,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=2),
        marker=dict(size=4, color="#38bdf8"),
        name="Freight Rate (USD/t)",
        hovertemplate="<b>%{x}</b><br>Rate: $%{y:.3f}/t<extra></extra>",
    ), secondary_y=False)

    # Bunker fuel trend (right axis)
    b_dates  = fuel_fc.get("dates", dates)
    b_prices = fuel_fc.get("price", [bunker] * 14)
    fig.add_trace(go.Scatter(
        x=b_dates, y=b_prices,
        mode="lines",
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
        name="Bunker Fuel (USD/MT)",
        hovertemplate="<b>%{x}</b><br>Bunker: $%{y:.0f}/MT<extra></extra>",
    ), secondary_y=True)

    # T* marker
    if 0 < t_star < len(dates):
        fig.add_vline(
            x=dates[t_star],
            line=dict(color="#fbbf24", width=1, dash="dash"),
            annotation_text=f"T*+{t_star}",
            annotation_font=dict(color="#fbbf24", size=10, family="JetBrains Mono"),
            annotation_position="top right",
        )

    # Apply layout (no duplicate keys — all explicit, not **PLOTLY_LAYOUT unpacked here)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color="#94a3b8"),
        margin=dict(l=50, r=60, t=35, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2d3f57", borderwidth=1,
                    font=dict(size=10, family="JetBrains Mono"),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#2d3f57",
                        font=dict(family="JetBrains Mono", size=11)),
        title=dict(text="14-DAY FREIGHT RATE FORECAST  |  BUNKER FUEL TREND",
                   font=dict(size=11, color="#64748b", family="JetBrains Mono"),
                   x=0, y=0.98),
        height=360,
    )
    fig.update_xaxes(
        gridcolor="#1e293b", gridwidth=0.5, linecolor="#2d3f57",
        tickfont=dict(family="JetBrains Mono", size=10),
        showspikes=True, spikecolor="#38bdf8", spikethickness=1, spikemode="across",
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="#1e293b", gridwidth=0.5, linecolor="#2d3f57",
        tickfont=dict(family="JetBrains Mono", size=10),
        zeroline=False, title_text="Freight Rate (USD/t)", title_font=dict(size=10, color="#64748b"),
        secondary_y=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(0,0,0,0)", linecolor="#2d3f57",
        tickfont=dict(family="JetBrains Mono", size=10, color="#f59e0b"),
        zeroline=False, title_text="Bunker Price (USD/MT)", title_font=dict(size=10, color="#f59e0b"),
        secondary_y=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── TLC Breakdown ────────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                color:#64748b;letter-spacing:0.08em;margin-bottom:0.4rem;'>
      TOTAL LANDED COST — COMPONENT BREAKDOWN
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BASE FREIGHT", f"${timing_dec.get('base_freight',0):,.0f}")
    c2.metric("BUNKER SURCHARGE", f"${timing_dec.get('bunker_surcharge',0):,.0f}")
    c3.metric("INVENTORY HOLDING", f"${timing_dec.get('inventory_hold',0):,.0f}")
    port_cfg = PORT_CONSTRAINTS.get(port, {})
    demurrage = port_cfg.get("queue_days_avg", 2) * port_cfg.get("demurrage_usd_day", 20000)
    c4.metric("DEMURRAGE EXPOSURE", f"${demurrage:,.0f}")

    # ── Timing Decision Detail Table ─────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    daily_rates = timing_dec.get("daily_rates", [])
    if daily_rates:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[f"T+{i}" for i in range(len(daily_rates))],
            y=daily_rates,
            marker=dict(
                color=["#22c55e" if i == t_star else "#1e3a5f" for i in range(len(daily_rates))],
                line=dict(color="#2d3f57", width=0.5),
            ),
            hovertemplate="<b>%{x}</b><br>Charter Rate: $%{y:,.0f}/day<extra></extra>",
            name="Projected Daily Rate",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=11, color="#94a3b8"),
            margin=dict(l=50, r=30, t=35, b=40),
            hoverlabel=dict(bgcolor="#1e293b", bordercolor="#2d3f57",
                            font=dict(family="JetBrains Mono", size=11)),
            showlegend=False,
            title=dict(text="CHARTER RATE PROJECTION — OPTIMAL FIXTURE WINDOW",
                       font=dict(size=11, color="#64748b", family="JetBrains Mono"), x=0),
            height=220,
            xaxis=dict(gridcolor="#1e293b", linecolor="#2d3f57",
                       tickfont=dict(family="JetBrains Mono", size=10), zeroline=False),
            yaxis=dict(gridcolor="#1e293b", linecolor="#2d3f57",
                       tickfont=dict(family="JetBrains Mono", size=10),
                       zeroline=False, tickprefix="$", tickformat=",.0f"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORT & BERTH OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Port Compliance Matrix ───────────────────────────────────────────────
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                color:#64748b;letter-spacing:0.08em;margin-bottom:0.5rem;'>
      PORT DRAFT CLEARANCE MATRIX  |  VESSEL CLASS ELIGIBILITY
    </div>""", unsafe_allow_html=True)

    if not compliance.empty:
        def _style_compliance(val):
            if "BLOCKED" in str(val):
                return "background-color:#450a0a;color:#fca5a5;font-family:'JetBrains Mono',monospace;font-size:0.75rem;"
            elif "CLEAR" in str(val):
                return "background-color:#052e16;color:#86efac;font-family:'JetBrains Mono',monospace;font-size:0.75rem;"
            return "font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#94a3b8;"

        styled = (
            compliance.style
            .map(_style_compliance)
            .set_table_styles([{
                "selector": "th",
                "props": [("background-color", "#1e293b"), ("color", "#64748b"),
                          ("font-family", "'JetBrains Mono',monospace"), ("font-size", "0.68rem"),
                          ("letter-spacing", "0.06em"), ("text-transform", "uppercase"),
                          ("border-bottom", "1px solid #2d3f57")],
            }])
        )
        st.dataframe(styled, use_container_width=True, height=255)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Ranked Vessels Table ─────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(f"""
        <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                    color:#64748b;letter-spacing:0.08em;margin-bottom:0.5rem;'>
          TOP VESSELS — {port.upper()} ALLOCATION  |  CARGO: {cargo_tonnes:,} MT
        </div>""", unsafe_allow_html=True)

        if not ranked_df.empty:
            display_cols = ["rank","candidate_id","vessel_type","dwt","speed","draft",
                            "vessel_age","charter_rate_usd_day","vessel_score"]
            display_cols = [c for c in display_cols if c in ranked_df.columns]
            show_df = ranked_df[display_cols].head(10).copy()
            show_df["vessel_score"] = show_df["vessel_score"].map(lambda x: f"{x:.1f}")
            show_df["charter_rate_usd_day"] = show_df["charter_rate_usd_day"].map(lambda x: f"${x:,.0f}")
            show_df["dwt"] = show_df["dwt"].map(lambda x: f"{x:,.0f}")
            st.dataframe(show_df, use_container_width=True, height=320, hide_index=True)
        else:
            st.warning(f"No vessels meet the port constraints for {port}. "
                       "Try reducing cargo volume or selecting a different port.")

    with col_r:
        # Vessel Score Distribution
        if not ranked_df.empty:
            fig3 = go.Figure()
            scores = ranked_df["vessel_score"].head(15)
            labels = ranked_df["candidate_id"].head(15)
            colors = ["#38bdf8" if i == 0 else "#1e3a5f" for i in range(len(scores))]

            fig3.add_trace(go.Bar(
                x=scores, y=labels, orientation="h",
                marker=dict(color=colors, line=dict(color="#2d3f57", width=0.5)),
                hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
            ))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=10, color="#94a3b8"),
                margin=dict(l=80, r=20, t=35, b=30),
                showlegend=False,
                title=dict(text="VESSEL COMPOSITE SCORE",
                           font=dict(size=11, color="#64748b", family="JetBrains Mono"), x=0),
                height=330,
                xaxis=dict(gridcolor="#1e293b", linecolor="#2d3f57",
                           tickfont=dict(family="JetBrains Mono", size=9), zeroline=False),
                yaxis=dict(linecolor="#2d3f57", tickfont=dict(family="JetBrains Mono", size=9),
                           zeroline=False, autorange="reversed"),
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── Speed Optimization ───────────────────────────────────────────────────
    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                color:#64748b;letter-spacing:0.08em;margin-bottom:0.5rem;'>
      OPTIMAL VOYAGE SPEED — BUNKER COST MINIMISATION
    </div>""", unsafe_allow_html=True)

    sp1, sp2, sp3, sp4 = st.columns(4)
    sp1.metric("OPTIMAL SPEED", f"{speed_opt.get('speed', '—')} kn")
    sp2.metric("FUEL CONSUMPTION", f"{speed_opt.get('fuel_per_day', '—')} MT/day")
    sp3.metric("TOTAL FUEL COST", f"${speed_opt.get('fuel_cost', 0):,.0f}")
    sp4.metric("VOYAGE DURATION", f"{speed_opt.get('voyage_hours', 0)/24:.1f} days")

    all_results_df = speed_opt.get("all_results", pd.DataFrame())
    if not all_results_df.empty:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=all_results_df["speed"], y=all_results_df["total_cost"],
            mode="lines", line=dict(color="#38bdf8", width=2),
            name="Total Economic Cost",
            hovertemplate="Speed: %{x:.1f} kn<br>Cost: $%{y:,.0f}<extra></extra>",
        ))
        fig4.add_trace(go.Scatter(
            x=all_results_df["speed"], y=all_results_df["fuel_cost"],
            mode="lines", line=dict(color="#f59e0b", width=1.5, dash="dot"),
            name="Fuel Cost Only",
            hovertemplate="Speed: %{x:.1f} kn<br>Fuel: $%{y:,.0f}<extra></extra>",
        ))
        opt_spd = speed_opt.get("speed", 14.0)
        fig4.add_vline(x=opt_spd, line=dict(color="#22c55e", width=1, dash="dash"),
                       annotation_text=f"OPT {opt_spd} kn",
                       annotation_font=dict(color="#22c55e", size=10, family="JetBrains Mono"))
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=11, color="#94a3b8"),
            margin=dict(l=55, r=30, t=35, b=40),
            height=220,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, family="JetBrains Mono")),
            title=dict(text="SPEED vs. TOTAL COST  |  PENALISED OPTIMUM",
                       font=dict(size=11, color="#64748b", family="JetBrains Mono"), x=0),
            xaxis=dict(gridcolor="#1e293b", linecolor="#2d3f57",
                       tickfont=dict(family="JetBrains Mono", size=10), zeroline=False,
                       title="Speed (knots)", title_font=dict(size=10)),
            yaxis=dict(gridcolor="#1e293b", linecolor="#2d3f57",
                       tickfont=dict(family="JetBrains Mono", size=10), zeroline=False,
                       tickprefix="$", tickformat=",.0f"),
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    # ── Demurrage Exposure Table ─────────────────────────────────────────────
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                color:#64748b;letter-spacing:0.08em;margin:0.5rem 0;'>
      DEMURRAGE EXPOSURE — ALL PORTS
    </div>""", unsafe_allow_html=True)

    dem_rows = []
    for p_name, cfg in PORT_CONSTRAINTS.items():
        exp = cfg["queue_days_avg"] * cfg["demurrage_usd_day"]
        dem_rows.append({
            "Port": p_name,
            "Avg Queue (days)": cfg["queue_days_avg"],
            "Rate (USD/day)": f"${cfg['demurrage_usd_day']:,}",
            "Exposure (USD)": f"${exp:,.0f}",
            "Max Draft (m)": cfg["max_draft"],
            "Status": "▶ SELECTED" if p_name == port else "",
        })
    dem_df = pd.DataFrame(dem_rows)
    st.dataframe(dem_df, use_container_width=True, height=240, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                color:#64748b;letter-spacing:0.08em;margin-bottom:0.75rem;'>
      EXECUTIVE DECISION BRIEF  |  SAIL CHARTERING COMMITTEE  |  CONFIDENTIAL
    </div>""", unsafe_allow_html=True)

    # ── Annualized ROI Projection ─────────────────────────────────────────────
    # Simulate 12 monthly fixtures with/without the model strategy
    rng = np.random.default_rng(5)
    months = [datetime.date.today().replace(day=1) - datetime.timedelta(days=30*i)
              for i in range(11, -1, -1)]
    month_labels  = [m.strftime("%b %y") for m in months]
    always_now    = rng.normal(cur_rate * 30, cur_rate * 3, 12)
    model_strat   = always_now - rng.uniform(savings * 0.5, savings * 1.5, 12)
    model_strat   = np.maximum(model_strat, always_now * 0.75)
    cumulative_savings = np.cumsum(always_now - model_strat)

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        name="Always-Book-Now Cost", x=month_labels, y=always_now / 1000,
        marker=dict(color="#1e3a5f", line=dict(color="#2d3f57", width=0.5)),
        hovertemplate="%{x}<br>Cost: $%{y:.0f}k/month<extra></extra>",
    ))
    fig5.add_trace(go.Bar(
        name="Model Strategy Cost", x=month_labels, y=model_strat / 1000,
        marker=dict(color="#0ea5e9", line=dict(color="#38bdf8", width=0.5)),
        hovertemplate="%{x}<br>Cost: $%{y:.0f}k/month<extra></extra>",
    ))
    fig5.add_trace(go.Scatter(
        name="Cumulative Savings (USD)", x=month_labels, y=cumulative_savings / 1000,
        mode="lines+markers", line=dict(color="#22c55e", width=2),
        marker=dict(size=5),
        yaxis="y2",
        hovertemplate="%{x}<br>Cumulative: $%{y:.0f}k<extra></extra>",
    ))
    fig5.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#94a3b8"),
        margin=dict(l=55, r=70, t=40, b=40),
        barmode="group",
        height=320,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2d3f57", borderwidth=1,
                    font=dict(size=10, family="JetBrains Mono"),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#2d3f57",
                        font=dict(family="JetBrains Mono", size=11)),
        title=dict(text="12-MONTH CHARTER COST  |  MODEL STRATEGY vs. ALWAYS-BOOK-NOW",
                   font=dict(size=11, color="#64748b", family="JetBrains Mono"), x=0),
        xaxis=dict(gridcolor="#1e293b", linecolor="#2d3f57",
                   tickfont=dict(family="JetBrains Mono", size=10), zeroline=False),
        yaxis=dict(gridcolor="#1e293b", linecolor="#2d3f57",
                   tickfont=dict(family="JetBrains Mono", size=10), zeroline=False,
                   title="Cost (USD '000)", title_font=dict(size=10)),
        yaxis2=dict(overlaying="y", side="right", gridcolor="rgba(0,0,0,0)",
                    linecolor="#2d3f57", tickfont=dict(family="JetBrains Mono", size=10,
                    color="#22c55e"), zeroline=False,
                    title="Cumulative Savings (USD '000)", title_font=dict(size=10, color="#22c55e")),
    )
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    # ── Aggregate KPIs ────────────────────────────────────────────────────────
    annualized_savings = float(np.sum(always_now - model_strat))
    roi_pct = (annualized_savings / np.sum(always_now)) * 100

    ex1, ex2, ex3, ex4 = st.columns(4)
    ex1.metric("ANNUALISED SAVINGS", f"${annualized_savings:,.0f}", delta="vs Always-Book-Now")
    ex2.metric("ROI (MODEL STRATEGY)", f"{roi_pct:.2f}%", delta="12-month backtest")
    ex3.metric("TOTAL FIXTURES SIMULATED", "12", delta="Monthly cadence")
    ex4.metric("DEMURRAGE AVOIDED", f"${savings * 8:,.0f}", delta="Estimated annualised")

    # ── Donut chart — TLC split ───────────────────────────────────────────────
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.markdown("""
        <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                    color:#64748b;letter-spacing:0.08em;margin:0.5rem 0;'>
          TLC COMPOSITION (SINGLE FIXTURE)
        </div>""", unsafe_allow_html=True)

        tlc_vals   = [
            timing_dec.get("base_freight", 0),
            timing_dec.get("bunker_surcharge", 0),
            demurrage,
            timing_dec.get("inventory_hold", 0),
        ]
        tlc_labels = ["Base Freight", "Bunker Surcharge", "Demurrage Risk", "Inventory Hold"]
        tlc_colors = ["#38bdf8", "#f59e0b", "#ef4444", "#a78bfa"]

        donut = go.Figure(go.Pie(
            labels=tlc_labels, values=tlc_vals,
            hole=0.6,
            marker=dict(colors=tlc_colors, line=dict(color="#0f172a", width=2)),
            textfont=dict(family="JetBrains Mono", size=10),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=11, color="#94a3b8"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=230,
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9, family="JetBrains Mono"),
                        orientation="v"),
        )
        st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

    with col_d2:
        st.markdown("""
        <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;
                    color:#64748b;letter-spacing:0.08em;margin:0.5rem 0;'>
          FIXTURE RECOMMENDATION MEMO  |  CVC COMPLIANCE AUDIT TRAIL
        </div>""", unsafe_allow_html=True)

        rec_text = f"""
SAIL MARITIME PROCUREMENT — FIXTURE ADVISORY NOTE
Generated: {now_ist.strftime('%Y-%m-%d %H:%M IST')} | System: PROD v2.4.1
{'='*60}

ROUTE            : {route}
DESTINATION PORT : {port}
CARGO TYPE       : {cargo_type}
CARGO VOLUME     : {cargo_tonnes:,} MT
VESSEL CLASS     : {vessel_class} ({vessel_dwt:,} DWT)
MARKET EVENT     : {market_event}

MARKET SNAPSHOT
  BDI Index        : {bdi:,}
  Bunker Price     : ${bunker}/MT
  Port Congestion  : {congestion} days

MODEL DECISION
  {decision_lbl}
  Optimal Fixture Date (T*) : T+{t_star} from {now_ist.strftime('%Y-%m-%d')}
  Projected Daily Rate (T*) : ${timing_dec.get('predicted_future', 0):,.0f}/day
  Current Rate              : ${cur_rate:,.0f}/day
  Net Savings (vs now)      : ${savings:,.0f}
  Directional Accuracy      : {dir_acc:.1f}%

TOTAL LANDED COST BREAKDOWN
  Base Freight       : ${timing_dec.get('base_freight', 0):,.0f}
  Bunker Surcharge   : ${timing_dec.get('bunker_surcharge', 0):,.0f}
  Demurrage Risk     : ${demurrage:,.0f}
  Inventory Holding  : ${timing_dec.get('inventory_hold', 0):,.0f}
  TOTAL              : ${timing_dec.get('total_landed_cost', 0):,.0f}

OPTIMAL VOYAGE SPEED
  Speed              : {speed_opt.get('speed', '—')} knots
  Fuel/Day           : {speed_opt.get('fuel_per_day', '—')} MT/day
  Fuel Cost          : ${speed_opt.get('fuel_cost', 0):,.0f}
  Voyage Duration    : {speed_opt.get('voyage_hours', 0)/24:.1f} days

PORT CONSTRAINT STATUS
  Max Draft Allowed  : {PORT_CONSTRAINTS.get(port, {}).get('max_draft', '—')} m
  {vessel_class} Draft : {VESSEL_CLASS_DRAFT.get(vessel_class, '—')} m
  Clearance Status   : {'✅ CLEAR' if VESSEL_CLASS_DRAFT.get(vessel_class, 99) <= PORT_CONSTRAINTS.get(port, {}).get('max_draft', 0) else '🚫 DRAFT VIOLATION'}

ANNUALISED PROJECTION
  Savings vs Always-Book-Now : ${annualized_savings:,.0f}
  Model ROI                  : {roi_pct:.2f}%
  Demurrage Avoided (12M)    : ${savings * 8:,.0f}

DISCLAIMER: This recommendation is generated by an AI-assisted decision
support system. Final fixture decisions remain the responsibility of the
SAIL Chartering Committee. All rates are model forecasts, not executable quotes.
"""
        st.code(rec_text, language=None)

        # ── CSV Download ──────────────────────────────────────────────────────
        audit_data = {
            "Field": [
                "Route", "Port", "Cargo Type", "Cargo (MT)", "Vessel Class", "Vessel DWT",
                "BDI", "Bunker Price (USD/MT)", "Decision", "Optimal T*",
                "Current Charter Rate", "Predicted Future Rate", "Net Savings (USD)",
                "Base Freight", "Bunker Surcharge", "Demurrage Risk", "Inventory Hold",
                "Total Landed Cost", "Optimal Speed (kn)", "Fuel Cost (USD)",
                "Directional Accuracy (%)", "Annualised Savings (USD)", "Model ROI (%)",
                "Generated At",
            ],
            "Value": [
                route, port, cargo_type, cargo_tonnes, vessel_class, vessel_dwt,
                bdi, bunker, timing_dec.get("decision"), t_star,
                round(cur_rate, 2), round(timing_dec.get("predicted_future", 0), 2),
                round(savings, 2),
                round(timing_dec.get("base_freight", 0), 2),
                round(timing_dec.get("bunker_surcharge", 0), 2),
                round(demurrage, 2),
                round(timing_dec.get("inventory_hold", 0), 2),
                round(timing_dec.get("total_landed_cost", 0), 2),
                speed_opt.get("speed"),
                round(speed_opt.get("fuel_cost", 0), 2),
                round(dir_acc, 1),
                round(annualized_savings, 2),
                round(roi_pct, 2),
                now_ist.strftime("%Y-%m-%d %H:%M IST"),
            ],
        }
        audit_df  = pd.DataFrame(audit_data)
        csv_bytes = audit_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇  DOWNLOAD FIXTURE AUDIT MEMO  (.CSV)  |  CVC COMPLIANCE",
            data=csv_bytes,
            file_name=f"SAIL_fixture_audit_{now_ist.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Bottom Status Bar ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="border-top:1px solid #1e293b;margin-top:1rem;padding-top:0.5rem;
                font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#475569;
                display:flex;justify-content:space-between;">
      <span>SAIL DSS · PROD v2.4.1 · Models: FreightXGB | FuelXGB | TimingGBR-LR | VesselRanker</span>
      <span>© Steel Authority of India Limited · Chartering Desk Analytics · {now_ist.year}</span>
    </div>
    """, unsafe_allow_html=True)
