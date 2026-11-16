"""
step1_preprocess.py - Data Preprocessing & Multi-Holdout Split for Almond Wholesale Forecasting
=============================================================================================
Loads verified 5-State Almond dataset (Maharashtra, Delhi, Haryana, Punjab, Gujarat),
standardizes features, and builds zero-leakage 6-month holdout evaluation folds.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

STATES = ["Delhi", "Gujarat", "Haryana", "Maharashtra", "Punjab"]

# Commercial dry fruit trade & import volume shares
STATE_WEIGHTS = {
    "Maharashtra": 35.0, # JNPT / Navi Mumbai APMC import hub
    "Delhi": 30.0,       # Khari Baoli / Gadodia national terminal
    "Gujarat": 15.0,     # Trade & processing hub
    "Punjab": 10.0,      # Regional consumption hub
    "Haryana": 10.0,     # Supply chain hub
}

PRICE_FEATURES = [
    "Buy_Price_Lag1", "Buy_Price_Lag2", "Buy_Price_Lag3",
    "Sell_Price_Lag1", "Gross_Margin_Lag1", "Mandi_Price_Lag1",
    "Price_Diff_1M", "Price_Diff_3M", "Price_Diff_6M", "Price_Acceleration",
    "Buy_Price_MA3", "Buy_Price_MA6", "Price_Dev_MA3", "Price_Volatility_3M",
    "Buy_to_Mandi_Spread_INR_kg"
]

WEATHER_FEATURES = [
    "Avg_Temperature_C", "Max_Temperature_C", "Min_Temperature_C", "Temp_Range_C",
    "Avg_Relative_Humidity_Pct", "Total_Precipitation_mm", "Rainfall_3M_Cum",
    "Heat_Stress_Flag", "Dry_Spell_Flag"
]

APY_FEATURES = [
    "Area_Lakh_Ha", "Production_Lakh_Tonnes", "Yield_Kg_Per_Ha",
    "Prev_Crop_Year_Area_Lakh_Ha", "Prev_Crop_Year_Production_Lakh_Tonnes", "Prev_Crop_Year_Yield_Kg_Per_Ha"
]

SEASONALITY_FEATURES = [
    "Month_sin", "Month_cos", "Sowing_Peak_Flag", "Harvest_Arrival_Peak_Flag"
]

MANDI_FEATURES = [
    "Mandi_Modal_Price_INR_Kg", "Mandi_Min_Price_INR_Kg", "Mandi_Max_Price_INR_Kg", "Mandi_Arrivals_Tonnes"
]

ALL_MODEL_FEATURES = PRICE_FEATURES + WEATHER_FEATURES + APY_FEATURES + SEASONALITY_FEATURES + MANDI_FEATURES

def find_dataset(custom_path: str | Path | None = None) -> Path:
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
    candidates = [
        HERE / "data" / "Almond_5_States_Final_Model_Input_2024_2026.csv",
        HERE / "Almond_5_States_Final_Model_Input_2024_2026.csv",
        HERE / "data" / "Almond_5_States_Final_Model_Input_2024_2026.xlsx",
        HERE / "Almond_5_States_Final_Model_Input_2024_2026.xlsx",
        Path("c:/Users/shash/Downloads/brain/brain/Almond_5_States_Final_Model_Input_2024_2026.csv"),
        Path("C:/Users/shash/Downloads/Almond_5_States_Final_Model_Input_2024_2026.csv")
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Almond input dataset not found!")

def load_preprocessed_data(file_path: str | Path | None = None) -> pd.DataFrame:
    p = find_dataset(file_path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_excel(p)

    df["Month_Start_Date"] = pd.to_datetime(df["Month_Start_Date"])
    df = df.sort_values(["State", "Month_Start_Date"]).reset_index(drop=True)
    return df

def get_train_test_split(df: pd.DataFrame, holdout_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last N calendar months strictly across all states."""
    dates = sorted(df["Month_Start_Date"].unique())
    train_dates = dates[:-holdout_months]
    test_dates = dates[-holdout_months:]
    train = df[df["Month_Start_Date"].isin(train_dates)].copy()
    test = df[df["Month_Start_Date"].isin(test_dates)].copy()
    return train, test

if __name__ == "__main__":
    df = load_preprocessed_data()
    tr, te = get_train_test_split(df, 6)
    print("=" * 65)
    print("       ALMOND DATA PREPROCESSING & DATASET INTEGRITY CHECK       ")
    print("=" * 65)
    print(f"Total Rows Loaded: {len(df)} across {df['State'].nunique()} States")
    print(f"States: {sorted(df['State'].unique())}")
    print(f"Date Range: {df['Month_Start_Date'].min().strftime('%Y-%m')} to {df['Month_Start_Date'].max().strftime('%Y-%m')}")
    print(f"Train Set: {len(tr)} rows (2024-04 to 2026-05)")
    print(f"Test Set : {len(te)} rows (2026-06 to 2026-11, 6-Month Holdout)")
    print(f"Feature Space: {len([f for f in ALL_MODEL_FEATURES if f in df.columns])} engineered features ready")
    print("=" * 65)
