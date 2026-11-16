"""
app.py - Almond Wholesale Price & Margin Forecasting Dashboard (Streamlit)
==========================================================================
Presents:
- 6-Month Horizon Price & Margin Forecasts (Dec 2026 - May 2027)
- Baseline vs. Market-Speculation Shock Scenarios
- State-wise Wholesale Buying vs. Selling Breakdown
- Active Market Speculation Intelligence & Policy Catalysts
(Model backtest evaluation metrics excluded per executive user directive)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from shock_engine import apply_market_shocks, load_speculations

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Almond 6-Month Price & Margin Forecast", layout="wide", page_icon="🌰")

st.title("🌰 Almond Wholesale Price & Margin Forecasting Dashboard")
st.markdown("### 6-Month Horizon (December 2026 — May 2027) Across 5 Key Commercial States")

# Load data
fc_csv = HERE / "final_almond_forecast_6m.csv"
if not fc_csv.exists():
    from step3_final_forecast import generate_forecasts
    df_fc = generate_forecasts()
else:
    df_fc = pd.read_csv(fc_csv)

# Sidebar filters
st.sidebar.header("🕹️ Scenario & View Controls")
state_sel = st.sidebar.selectbox("Select State", ["All States (Aggregate)"] + sorted(df_fc["State"].unique()))
scenario_sel = st.sidebar.radio("Market Price Scenario", ["Shock-Adjusted (Speculation Active)", "Baseline (Statistical Only)"])

if state_sel != "All States (Aggregate)":
    view_df = df_fc[df_fc["State"] == state_sel].copy()
else:
    view_df = df_fc.groupby("Month_Start_Date", as_index=False).agg({
        "Baseline_Buy_Price_INR_kg": "mean",
        "Baseline_Sell_Price_INR_kg": "mean",
        "Baseline_Gross_Margin_INR_kg": "mean",
        "Baseline_Gross_Margin_Pct": "mean",
        "Shocked_Buy_Price_INR_kg": "mean",
        "Shocked_Sell_Price_INR_kg": "mean",
        "Shocked_Gross_Margin_INR_kg": "mean",
        "Shocked_Gross_Margin_Pct": "mean",
        "Buy_Price_Shock_Pct": "mean",
        "Sell_Price_Shock_Pct": "mean"
    })
    view_df["State"] = "All States"

bp_col = "Shocked_Buy_Price_INR_kg" if "Shock" in scenario_sel else "Baseline_Buy_Price_INR_kg"
sp_col = "Shocked_Sell_Price_INR_kg" if "Shock" in scenario_sel else "Baseline_Sell_Price_INR_kg"
gm_col = "Shocked_Gross_Margin_INR_kg" if "Shock" in scenario_sel else "Baseline_Gross_Margin_INR_kg"
gmp_col = "Shocked_Gross_Margin_Pct" if "Shock" in scenario_sel else "Baseline_Gross_Margin_Pct"

# Key Metric Cards
c1, c2, c3, c4 = st.columns(4)
avg_bp = view_df[bp_col].mean()
avg_sp = view_df[sp_col].mean()
avg_gm = view_df[gm_col].mean()
avg_gmp = view_df[gmp_col].mean()

c1.metric("Avg Forecast Buying Price", f"₹{avg_bp:.2f} /kg")
c2.metric("Avg Forecast Selling Price", f"₹{avg_sp:.2f} /kg")
c3.metric("Avg Forecast Gross Margin", f"₹{avg_gm:.2f} /kg")
c4.metric("Avg Forecast Margin %", f"{avg_gmp:.1f} %")

st.markdown("---")

# Visual Forecast Chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=view_df["Month_Start_Date"], y=view_df[bp_col],
    mode="lines+markers", name=f"Buying Price ({scenario_sel[:8]})",
    line=dict(color="#1f77b4", width=3)
))
fig.add_trace(go.Scatter(
    x=view_df["Month_Start_Date"], y=view_df[sp_col],
    mode="lines+markers", name=f"Selling Price ({scenario_sel[:8]})",
    line=dict(color="#2ca02c", width=3)
))
fig.add_trace(go.Bar(
    x=view_df["Month_Start_Date"], y=view_df[gm_col],
    name="Gross Margin (₹/kg)", marker_color="#ff7f0e", opacity=0.4
))
fig.update_layout(
    title=f"6-Month Price & Margin Trajectory for {state_sel}",
    xaxis_title="Forecast Month", yaxis_title="INR per kg",
    hovermode="x unified", height=450
)
st.plotly_chart(fig, use_container_width=True)

# Table preview
st.subheader("📋 Detailed Monthly Forecast Values")
display_cols = ["Month_Start_Date", "State", bp_col, sp_col, gm_col, gmp_col]
if "Shock" in scenario_sel:
    display_cols += ["Buy_Price_Shock_Pct", "Sell_Price_Shock_Pct"]

st.dataframe(view_df[display_cols].style.format({
    bp_col: "₹{:.2f}", sp_col: "₹{:.2f}", gm_col: "₹{:.2f}",
    gmp_col: "{:.1f}%",
    "Buy_Price_Shock_Pct": "{:+.1f}%", "Sell_Price_Shock_Pct": "{:+.1f}%"
}), use_container_width=True)

# Market Speculation Intelligence Section
st.markdown("---")
st.subheader("🌐 Active Market Speculations & Global Catalysts")
spec_factors = load_speculations()
for fac in spec_factors:
    with st.expander(f"📌 {fac['name']} (Active: {', '.join(fac['active_months'])})"):
        st.markdown(f"**Impact:** Buying Price: `{fac['buying_price_shock_pct']:+.1f}%` | Selling Price: `{fac['selling_price_shock_pct']:+.1f}%`")
        st.markdown(f"**Analysis:** {fac['rationale']}")
