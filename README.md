# Almond 6-Month Wholesale Price & Margin Forecasting Pipeline

## Overview
Automated machine learning forecasting system for **Almonds** across 5 major commercial trade states in India:
- **Maharashtra (MH)**: Primary import terminal & JNPT / Navi Mumbai APMC port market
- **Delhi (DL)**: Khari Baoli / Gadodia wholesale national redistribution terminal
- **Gujarat (GJ)**: Processing, commercial distribution, and confectionery hub
- **Punjab (PB)**: Northwest consumption and retail market hub
- **Haryana (HR)**: Regional logistics and consumer market hub

## Execution Sequence
Run the pipeline in sequence using any of the following commands:
```bash
# Step 1: Preprocess and validate 160-row dataset
python step1_preprocess.py       # (or python 01_merge_internal_external.py)

# Step 2: Multi-model training and 6-month holdout backtesting
python step2_backtest.py         # (or python 02_train_evaluate_models.py)

# Step 3: Generate 6-month recursive forecast (Dec 2026 - May 2027) & apply market shocks
python step3_final_forecast.py   # (or python 03_forecast_future.py)

# Step 4: Launch interactive Streamlit dashboard
streamlit run app.py
```
Or double-click `run_pipeline.bat` on Windows.
