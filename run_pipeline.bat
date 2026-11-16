@echo off
echo ========================================================
echo       ALMOND PRICE FORECASTING PIPELINE (5 STATES)
echo ========================================================
echo [1/3] Preprocessing and feature validation...
python step1_preprocess.py
echo.
echo [2/3] Multi-model training and holdout backtesting...
python step2_backtest.py
echo.
echo [3/3] Generating 6-month recursive forecast and market shocks...
python step3_final_forecast.py
echo.
echo ========================================================
echo Pipeline execution complete!
echo To launch the interactive dashboard, run:
echo   streamlit run app.py
echo ========================================================
pause
