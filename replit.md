# Flowrate Forecasting App

## Overview
A Streamlit web application for analyzing and forecasting water flowrate based on historical flowrate and weather data from Open-Meteo API (precipitation, ET0 evapotranspiration). Supports XGBoost, ML ensemble, and SARIMAX models. Multi-site support with Site B and Site K.

## Recent Changes
- 2026-02-20: Added multi-site support (Site B + Site K) with site selector
- 2026-02-20: Replaced st.tabs with st.radio for persistent tab navigation
- 2026-02-20: Removed Prophet and LSTM, kept XGBoost/ML/SARIMAX only
- 2026-02-20: Fixed tab reset bug by using st.form for model controls
- 2026-02-20: Simplified to 2 weather variables (precipitation, ET0)
- 2026-02-20: Combined weather charts into single dual-axis visualization
- 2026-02-20: Integrated Open-Meteo API for weather data
- 2026-02-20: Added confidence level analysis, multiple data input, rainfall lag in SARIMAX

## Project Architecture
- `app.py` - Main Streamlit application with all logic
- `attached_assets/` - Contains TSV data files per site
  - Site B: `Pasted-date-flowrate-rainfall-...txt` (lat: 3.238704, lon: 98.526123238704)
  - Site K: `site_k_flowrate.txt` (lat: -6.752897, lon: 106.746108)
- `added_data_site_b.json` / `added_data_site_k.json` - Persistent added data per site
- `.streamlit/config.toml` - Streamlit server configuration (port 5000)

## Sites
- **Site B** - lat: 3.238704, lon: 98.526123238704
- **Site K** - lat: -6.752897, lon: 106.746108

## Data Sources
- Flowrate: From TSV files (user-provided, per site)
- Weather: Open-Meteo Archive API (per site coordinates)
  - precipitation_sum (monthly sum)
  - et0_fao_evapotranspiration (monthly sum)

## Models
1. **XGBoost** - Extreme Gradient Boosting with regularization and lagged features (default)
2. **ML (Ensemble RF+GBR)** - Random Forest + Gradient Boosting ensemble with lagged features
3. **SARIMAX** - Statistical model with seasonal ARIMA + exogenous weather variables

## Features
1. Multi-site support with site selector
2. Auto lag correlation analysis (weather variables -> flowrate, max 6 lags)
3. Auto stationarity check (ADF test) for all variables
4. Train vs Test visualization
5. Model selection: XGBoost / ML / SARIMAX
6. All models use weather variables with optimal lags
7. Confidence interval on forecasts + accuracy analysis
8. Export forecast results to CSV
9. Batch input new data points with auto weather fetch from Open-Meteo
10. Custom future weather assumptions for forecast scenarios
11. Persistent navigation using st.radio (no tab reset on rerun)

## Running
```bash
streamlit run app.py --server.port 5000
```
