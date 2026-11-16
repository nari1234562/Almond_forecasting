"""
step3_final_forecast.py - 6-Month Horizon Forecasting (Dec 2026 - May 2027) for Almond
=====================================================================================
Trains on full historical series and produces baseline and shock-adjusted forecasts
across 5 key states (Maharashtra, Delhi, Haryana, Punjab, Gujarat).
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from step1_preprocess import load_preprocessed_data, ALL_MODEL_FEATURES, STATES
from shock_engine import apply_market_shocks

HERE = Path(__file__).resolve().parent

FORECAST_MONTHS = [
    "2026-12-01", "2027-01-01", "2027-02-01",
    "2027-03-01", "2027-04-01", "2027-05-01"
]

def generate_forecasts():
    df = load_preprocessed_data()
    features = [f for f in ALL_MODEL_FEATURES if f in df.columns]

    mask_b = df["Buy_Price_INR_kg"].notna()
    med_b = df[features].median()
    X_b = df.loc[mask_b, features].fillna(med_b)
    y_b = df.loc[mask_b, "Buy_Price_INR_kg"]

    mask_s = df["Sell_Price_INR_kg"].notna()
    X_s = df.loc[mask_s, features].fillna(med_b)
    y_s = df.loc[mask_s, "Sell_Price_INR_kg"]

    model_b = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 13)))
    model_s = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 13)))

    model_b.fit(X_b, y_b)
    model_s.fit(X_s, y_s)

    forecast_rows = []
    for st in STATES:
        st_df = df[df["State"] == st].sort_values("Month_Start_Date").copy()
        last_row = st_df.iloc[-1].to_dict()

        recent_bp = list(st_df["Buy_Price_INR_kg"].dropna().tail(6))
        recent_sp = list(st_df["Sell_Price_Filled_INR_kg"].dropna().tail(6))
        recent_mp = list(st_df["Mandi_Modal_Price_INR_Kg"].dropna().tail(6))

        for m_str in FORECAST_MONTHS:
            dt = pd.to_datetime(m_str)
            m_num = dt.month
            y_num = dt.year

            feat_vec = {}
            for f in features:
                feat_vec[f] = last_row.get(f, np.nan)

            feat_vec["Buy_Price_Lag1"] = recent_bp[-1]
            feat_vec["Buy_Price_Lag2"] = recent_bp[-2] if len(recent_bp) >= 2 else recent_bp[-1]
            feat_vec["Buy_Price_Lag3"] = recent_bp[-3] if len(recent_bp) >= 3 else recent_bp[-1]
            feat_vec["Sell_Price_Lag1"] = recent_sp[-1]
            feat_vec["Gross_Margin_Lag1"] = recent_sp[-1] - recent_bp[-1]
            feat_vec["Mandi_Price_Lag1"] = recent_mp[-1]

            feat_vec["Price_Diff_1M"] = recent_bp[-1] - recent_bp[-2] if len(recent_bp) >= 2 else 0.0
            feat_vec["Price_Diff_3M"] = recent_bp[-1] - recent_bp[-3] if len(recent_bp) >= 3 else 0.0
            feat_vec["Price_Diff_6M"] = recent_bp[-1] - recent_bp[-6] if len(recent_bp) >= 6 else 0.0
            feat_vec["Price_Acceleration"] = 0.0

            feat_vec["Buy_Price_MA3"] = np.mean(recent_bp[-3:])
            feat_vec["Buy_Price_MA6"] = np.mean(recent_bp[-6:])
            feat_vec["Price_Dev_MA3"] = recent_bp[-1] - feat_vec["Buy_Price_MA3"]
            feat_vec["Price_Volatility_3M"] = np.std(recent_bp[-3:], ddof=1) if len(recent_bp) >= 3 else 0.0
            feat_vec["Buy_to_Mandi_Spread_INR_kg"] = recent_bp[-1] - recent_mp[-1]

            feat_vec["Month_sin"] = np.sin(2 * np.pi * m_num / 12.0)
            feat_vec["Month_cos"] = np.cos(2 * np.pi * m_num / 12.0)

            X_curr = pd.DataFrame([feat_vec])[features].fillna(med_b)

            pred_bp = round(float(model_b.predict(X_curr)[0]), 2)
            pred_sp = round(float(model_s.predict(X_curr)[0]), 2)

            # Commercial guard: SP must exceed BP by positive gross margin
            if pred_sp <= pred_bp:
                pred_sp = round(pred_bp * 1.08, 2)

            gm_rs = round(pred_sp - pred_bp, 2)
            gm_pct = round((gm_rs / pred_sp) * 100.0, 2)

            forecast_rows.append({
                "Month_Start_Date": m_str,
                "Year": y_num,
                "Month": dt.strftime("%b"),
                "Month_Num": m_num,
                "State": st,
                "Commodity": "Almond",
                "Baseline_Buy_Price_INR_kg": pred_bp,
                "Baseline_Sell_Price_INR_kg": pred_sp,
                "Baseline_Gross_Margin_INR_kg": gm_rs,
                "Baseline_Gross_Margin_Pct": gm_pct
            })

            recent_bp.append(pred_bp)
            recent_sp.append(pred_sp)
            recent_mp.append(pred_bp * 0.96)

    df_base = pd.DataFrame(forecast_rows)
    df_shocked = apply_market_shocks(df_base)

    out_csv = HERE / "final_almond_forecast_6m.csv"
    out_xlsx = HERE / "final_almond_forecast_6m.xlsx"
    df_shocked.to_csv(out_csv, index=False)
    df_shocked.to_excel(out_xlsx, index=False)

    print("=" * 68)
    print("      ALMOND 6-MONTH RECURSIVE FORECAST (DEC 2026 - MAY 2027)     ")
    print("=" * 68)
    print(f"Generated {len(df_shocked)} forecast rows across {df_shocked['State'].nunique()} states.")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_xlsx}")
    print("=" * 68)

    return df_shocked

if __name__ == "__main__":
    generate_forecasts()
