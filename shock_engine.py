"""
shock_engine.py - Market Shock Engine for Almond Wholesale Forecasting
=====================================================================
Applies domain-verified external market speculation shocks onto
baseline statistical forecasts for the 6-month horizon (Dec 2026 - May 2027).
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

HERE = Path(__file__).resolve().parent

def load_speculations(json_path: Path | str | None = None) -> List[Dict]:
    path = Path(json_path) if json_path else HERE / "external_market_speculations.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("market_factors", [])

def compute_monthly_shocks(months: List[str], json_path: Path | str | None = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    factors = load_speculations(json_path)
    buy_shocks = {m: 0.0 for m in months}
    sell_shocks = {m: 0.0 for m in months}

    for fac in factors:
        b_pct = fac.get("buying_price_shock_pct", 0.0) / 100.0
        s_pct = fac.get("selling_price_shock_pct", 0.0) / 100.0
        for m in fac.get("active_months", []):
            if m in buy_shocks:
                buy_shocks[m] += b_pct
                sell_shocks[m] += s_pct

    return buy_shocks, sell_shocks

def apply_market_shocks(df_forecast: pd.DataFrame, json_path: Path | str | None = None) -> pd.DataFrame:
    df = df_forecast.copy()
    months_yyyymm = [str(d)[:7] for d in df["Month_Start_Date"].unique()]
    buy_shocks, sell_shocks = compute_monthly_shocks(months_yyyymm, json_path)

    df["Month_Key"] = df["Month_Start_Date"].astype(str).str[:7]
    df["Buy_Price_Shock_Pct"] = df["Month_Key"].map(buy_shocks).fillna(0.0) * 100.0
    df["Sell_Price_Shock_Pct"] = df["Month_Key"].map(sell_shocks).fillna(0.0) * 100.0

    df["Shocked_Buy_Price_INR_kg"] = (
        df["Baseline_Buy_Price_INR_kg"] * (1.0 + df["Buy_Price_Shock_Pct"] / 100.0)
    ).round(2)
    df["Shocked_Sell_Price_INR_kg"] = (
        df["Baseline_Sell_Price_INR_kg"] * (1.0 + df["Sell_Price_Shock_Pct"] / 100.0)
    ).round(2)

    df["Shocked_Gross_Margin_INR_kg"] = (
        df["Shocked_Sell_Price_INR_kg"] - df["Shocked_Buy_Price_INR_kg"]
    ).round(2)
    df["Shocked_Gross_Margin_Pct"] = (
        (df["Shocked_Gross_Margin_INR_kg"] / df["Shocked_Sell_Price_INR_kg"]) * 100.0
    ).round(2)

    df = df.drop(columns=["Month_Key"])
    return df
