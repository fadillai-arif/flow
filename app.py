import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import itertools
import openmeteo_requests
import requests_cache
from retry_requests import retry
import xgboost as xgb
import json
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Flowrate Forecasting App", layout="wide")

SITES = {
    "Site B": {
        "data_file": "attached_assets/Pasted-date-flowrate-rainfall-2016-01-37-8-84-02878607-2016-02_1771560863535.txt",
        "added_data_file": "added_data_site_b.json",
        "latitude": 3.238704,
        "longitude": 98.526123,
    },
    "Site T": {
        "data_file": "attached_assets/site_t_flowrate.txt",
        "added_data_file": "added_data_site_t.json",
        "latitude": -5.445672,
        "longitude": 104.668215,
        "has_water_level": True,
    },
    "Site L": {
        "data_file": "attached_assets/site_l_flowrate.txt",
        "added_data_file": "added_data_site_l.json",
        "latitude": -6.765598,
        "longitude": 106.827264,
    },
    "Site K": {
        "data_file": "attached_assets/site_k_flowrate.txt",
        "added_data_file": "added_data_site_k.json",
        "latitude": -6.752897,
        "longitude": 106.746108,
    },
    "Site S": {
        "data_file": "attached_assets/site_s_flowrate.txt",
        "added_data_file": "added_data_site_s.json",
        "latitude": -6.736265,
        "longitude": 107.686182,
        "has_water_level": True,
    },
    "Site KL": {
        "data_file": "attached_assets/site_kl_flowrate.txt",
        "added_data_file": "added_data_site_kl.json",
        "latitude": -7.53718,
        "longitude": 110.47984,
    },
    "Site M": {
        "data_file": "attached_assets/site_m_flowrate.txt",
        "added_data_file": "added_data_site_m.json",
        "latitude": -8.253142,
        "longitude": 115.304637,
        "has_water_level": True,
    },
}

WEATHER_VARS = ["precipitation", "et0"]
WEATHER_LABELS = {
    "precipitation": "Precipitation (mm)",
    "et0": "ET0 Evapotranspiration (mm)",
}


@st.cache_data(ttl=3600)
def fetch_openmeteo_monthly(start_date, end_date, latitude, longitude):
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["precipitation_sum", "et0_fao_evapotranspiration"],
        "timezone": "Asia/Bangkok",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_precipitation = daily.Variables(0).ValuesAsNumpy()
    daily_et0 = daily.Variables(1).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }
    daily_data["precipitation"] = daily_precipitation
    daily_data["et0"] = daily_et0

    daily_df = pd.DataFrame(data=daily_data)
    daily_df["date"] = daily_df["date"].dt.tz_localize(None)
    daily_df["year_month"] = daily_df["date"].dt.to_period("M")

    monthly = daily_df.groupby("year_month").agg(
        precipitation=("precipitation", "sum"),
        et0=("et0", "sum"),
    ).reset_index()
    monthly["date"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly.drop(columns=["year_month"])
    for col in WEATHER_VARS:
        monthly[col] = monthly[col].round(2)

    return monthly


@st.cache_data
def load_initial_data(data_file):
    df = pd.read_csv(data_file, sep="\t")
    # Robust date parsing to handle YYYY-MM and potential extra chars
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def load_data_with_weather(data_file, latitude, longitude):
    df = load_initial_data(data_file)
    start_date = df["date"].iloc[0].strftime("%Y-%m-%d")
    end_date_adj = (df["date"].iloc[-1] + pd.DateOffset(months=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    weather = fetch_openmeteo_monthly(start_date, end_date_adj, latitude, longitude)
    merged = pd.merge(df, weather, on="date", how="left")
    if "rainfall" in merged.columns:
        merged["precipitation"] = merged["precipitation"].fillna(merged["rainfall"])
    for col in WEATHER_VARS:
        merged[col] = merged[col].fillna(method="ffill")
    return merged


def load_added_data(added_data_file):
    if os.path.exists(added_data_file):
        try:
            with open(added_data_file, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_added_data(rows, added_data_file):
    with open(added_data_file, "w") as f:
        json.dump(rows, f, indent=2)


def get_data(site_config):
    df = load_data_with_weather(site_config["data_file"], site_config["latitude"], site_config["longitude"]).copy()
    added_rows = load_added_data(site_config["added_data_file"])
    if len(added_rows) > 0:
        added_df = pd.DataFrame(added_rows)
        added_df["date"] = pd.to_datetime(added_df["date"])
        for col in WEATHER_VARS:
            if col not in added_df.columns:
                added_df[col] = np.nan
        if "rainfall" not in added_df.columns:
            added_df["rainfall"] = added_df.get("precipitation", np.nan)
        df = pd.concat([df, added_df], ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)
        for col in WEATHER_VARS:
            df[col] = df[col].fillna(method="ffill")
        if "water_level" in df.columns:
            df["water level"] = df["water_level"].fillna(method="ffill")
    return df


def adf_test(series, name):
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "Variable": name,
        "ADF Statistic": round(result[0], 4),
        "p-value": round(result[1], 4),
        "Lags Used": result[2],
        "Observations": result[3],
        "Critical 1%": round(result[4]["1%"], 4),
        "Critical 5%": round(result[4]["5%"], 4),
        "Critical 10%": round(result[4]["10%"], 4),
        "Stationary": "Yes" if result[1] < 0.05 else "No",
    }


def find_best_lag(flowrate, variable, max_lag=6):
    n = len(flowrate)
    correlations = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(variable[:n], flowrate[:n])[0, 1]
        else:
            corr = np.corrcoef(variable[:n - lag], flowrate[lag:n])[0, 1]
        correlations.append({"Lag": lag, "Correlation": round(corr, 4)})
    corr_df = pd.DataFrame(correlations)
    best_lag = corr_df.loc[corr_df["Correlation"].abs().idxmax(), "Lag"]
    return int(best_lag), corr_df


def create_lagged_features(df, best_lags, n_lags=6):
    features = pd.DataFrame(index=df.index)
    for var in WEATHER_VARS:
        features[var] = df[var].values
        for i in range(1, n_lags + 1):
            features[f"{var}_lag{i}"] = df[var].shift(i)
    for i in range(1, n_lags + 1):
        features[f"flowrate_lag{i}"] = df["flowrate"].shift(i)
    for var in WEATHER_VARS:
        bl = best_lags.get(var, 0)
        if bl > 0:
            features[f"{var}_best_lag{bl}"] = df[var].shift(bl)
    features["month"] = df["date"].dt.month
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    return features


def auto_arima_search(train_series, max_p=3, max_d=2, max_q=3):
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            model = ARIMA(train_series, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except Exception:
            continue
    return best_order, best_aic


def auto_sarimax_search(train_series, exog_train, best_arima_order):
    p, d, q = best_arima_order
    best_aic = np.inf
    best_seasonal = (0, 0, 0, 12)
    seasonal_options = [
        (0, 0, 0, 12),
        (1, 0, 0, 12),
        (0, 0, 1, 12),
        (1, 0, 1, 12),
        (1, 1, 0, 12),
        (0, 1, 1, 12),
        (1, 1, 1, 12),
    ]
    for seasonal in seasonal_options:
        try:
            model = SARIMAX(
                train_series,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=200)
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_seasonal = seasonal
        except Exception:
            continue
    return best_seasonal, best_aic


def run_forecast(df, model_choice, test_size, forecast_months, future_weather=None, target_col="flowrate"):
    n = len(df)
    train_size = n - test_size
    if train_size < 2:
        train_size = 2
        test_size = n - train_size
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    train_target = train_df[target_col]
    test_target = test_df[target_col]

    results = {}

    if model_choice == "SARIMAX":
        best_order, _ = auto_arima_search(train_target)
        results["arima_order"] = best_order

        best_lags = {}
        for var in WEATHER_VARS:
            bl, _ = find_best_lag(df[target_col].values, df[var].values)
            best_lags[var] = bl
        results["weather_lags"] = str(best_lags)

        exog_cols = list(WEATHER_VARS)
        exog_df = df[WEATHER_VARS].copy()
        max_lag_val = max(best_lags.values()) if best_lags else 0
        for var in WEATHER_VARS:
            bl = best_lags[var]
            for lag in range(1, bl + 1):
                col_name = f"{var}_lag{lag}"
                exog_df[col_name] = df[var].shift(lag)
                exog_cols.append(col_name)

        valid_start = max_lag_val if max_lag_val > 0 else 0
        df_valid = df.iloc[valid_start:].reset_index(drop=True)
        exog_valid = exog_df.iloc[valid_start:].reset_index(drop=True)
        n_valid = len(df_valid)

        adj_test_size = min(test_size, n_valid - 2)
        if adj_test_size < 1: adj_test_size = 1
        
        train_size_s = n_valid - adj_test_size
        train_target_s = df_valid[target_col].iloc[:train_size_s]
        test_target_s = df_valid[target_col].iloc[train_size_s:]
        train_exog = exog_valid.iloc[:train_size_s].values
        test_exog = exog_valid.iloc[train_size_s:].values

        train_df = df_valid.iloc[:train_size_s]
        test_df = df_valid.iloc[train_size_s:]
        test_target = test_target_s

        best_seasonal, best_aic = auto_sarimax_search(train_target_s, train_exog, best_order)
        results["seasonal_order"] = best_seasonal
        results["aic"] = round(best_aic, 2)

        model = SARIMAX(
            train_target_s, exog=train_exog, order=best_order,
            seasonal_order=best_seasonal, enforce_stationarity=False, enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)
        test_pred = fitted.forecast(steps=adj_test_size, exog=test_exog)

        full_model = SARIMAX(
            df_valid[target_col], exog=exog_valid.values, order=best_order,
            seasonal_order=best_seasonal, enforce_stationarity=False, enforce_invertibility=False,
        )
        full_fitted = full_model.fit(disp=False, maxiter=200)

        fw = future_weather or {}
        future_exog_rows = []
        for step in range(forecast_months):
            row = []
            for var in WEATHER_VARS:
                val = fw.get(var, df[var].iloc[-1])
                row.append(val)
            for var in WEATHER_VARS:
                bl = best_lags[var]
                recent = list(df[var].values[-(bl):]) if bl > 0 else []
                for lag in range(1, bl + 1):
                    if step - lag >= 0:
                        row.append(fw.get(var, df[var].iloc[-1]))
                    elif len(recent) >= (lag - step):
                        row.append(recent[-(lag - step)])
                    else:
                        row.append(fw.get(var, df[var].iloc[-1]))
            future_exog_rows.append(row)
        future_exog = np.array(future_exog_rows)

        forecast_result = full_fitted.get_forecast(steps=forecast_months, exog=future_exog)
        forecast_vals = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)

    elif model_choice == "ML":
        best_lags = {}
        for var in WEATHER_VARS:
            bl, _ = find_best_lag(df[target_col].values, df[var].values)
            best_lags[var] = bl

        features = pd.DataFrame(index=df.index)
        for var in WEATHER_VARS:
            features[var] = df[var].values
            for i in range(1, 7):
                features[f"{var}_lag{i}"] = df[var].shift(i)
        for i in range(1, 7):
            features[f"target_lag{i}"] = df[target_col].shift(i)
        for var in WEATHER_VARS:
            bl = best_lags.get(var, 0)
            if bl > 0:
                features[f"{var}_best_lag{bl}"] = df[var].shift(bl)
        features["month"] = df["date"].dt.month
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
        
        target = df[target_col]

        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]

        offset = n - len(features)
        adj_train = max(train_size - offset, 2)
        adj_test = len(features) - adj_train
        if adj_test < 1:
            adj_train = len(features) - 1
            adj_test = 1
        if len(features) < 2:
             raise ValueError(f"Data tidak cukup untuk model ML pada {target_col}. Butuh minimal 2 baris data valid.")

        feat_train = features.iloc[:adj_train]
        feat_test = features.iloc[adj_train:]
        tgt_train = target.iloc[:adj_train]
        tgt_test = target.iloc[adj_train:]

        scaler = StandardScaler()
        feat_train_sc = scaler.fit_transform(feat_train)
        feat_test_sc = scaler.transform(feat_test)

        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.05)
        rf.fit(feat_train_sc, tgt_train)
        gb.fit(feat_train_sc, tgt_train)

        rf_pred = rf.predict(feat_test_sc)
        gb_pred = gb.predict(feat_test_sc)
        test_pred = pd.Series((rf_pred + gb_pred) / 2, index=tgt_test.index)
        test_target = tgt_test

        fw = future_weather or {}
        forecast_vals_list = []
        temp_targets = list(df[target_col].values[-6:])
        temp_weather = {var: list(df[var].values[-6:]) for var in WEATHER_VARS}

        for step in range(forecast_months):
            new_feat = pd.DataFrame(index=[0])
            for var in WEATHER_VARS:
                current_val = fw.get(var, df[var].iloc[-1])
                new_feat[var] = current_val
                for i in range(1, 7):
                    tw = temp_weather[var]
                    new_feat[f"{var}_lag{i}"] = tw[-i] if i <= len(tw) else current_val
            for i in range(1, 7):
                new_feat[f"target_lag{i}"] = temp_targets[-i] if i <= len(temp_targets) else df[target_col].iloc[-1]
            for var in WEATHER_VARS:
                bl = best_lags.get(var, 0)
                if bl > 0:
                    tw = temp_weather[var]
                    new_feat[f"{var}_best_lag{bl}"] = tw[-bl] if bl <= len(tw) else fw.get(var, df[var].iloc[-1])
            month_num = (df["date"].iloc[-1].month + step) % 12 + 1
            new_feat["month"] = month_num
            new_feat["month_sin"] = np.sin(2 * np.pi * month_num / 12)
            new_feat["month_cos"] = np.cos(2 * np.pi * month_num / 12)

            new_feat = new_feat[feat_train.columns]
            new_feat_sc = scaler.transform(new_feat)
            pred_val = (rf.predict(new_feat_sc)[0] + gb.predict(new_feat_sc)[0]) / 2
            forecast_vals_list.append(pred_val)

            temp_targets.append(pred_val)
            for var in WEATHER_VARS:
                temp_weather[var].append(fw.get(var, df[var].iloc[-1]))

        forecast_vals = pd.Series(forecast_vals_list)
        residuals = test_pred.values - test_target.values
        std_resid = np.std(residuals) if len(residuals) > 1 else 0.1
        lower = forecast_vals - 1.96 * std_resid
        upper = forecast_vals + 1.96 * std_resid
        conf_int = pd.DataFrame({"lower": lower.values, "upper": upper.values})

        results["weather_lags"] = str(best_lags)
        results["models"] = "RF + GBR Ensemble"

    elif model_choice == "XGBoost":
        best_lags = {}
        for var in WEATHER_VARS:
            bl, _ = find_best_lag(df[target_col].values, df[var].values)
            best_lags[var] = bl

        features = pd.DataFrame(index=df.index)
        for var in WEATHER_VARS:
            features[var] = df[var].values
            for i in range(1, 7):
                features[f"{var}_lag{i}"] = df[var].shift(i)
        for i in range(1, 7):
            features[f"target_lag{i}"] = df[target_col].shift(i)
        for var in WEATHER_VARS:
            bl = best_lags.get(var, 0)
            if bl > 0:
                features[f"{var}_best_lag{bl}"] = df[var].shift(bl)
        features["month"] = df["date"].dt.month
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

        target = df[target_col]

        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]

        offset = n - len(features)
        adj_train = max(train_size - offset, 2)
        adj_test = len(features) - adj_train
        if adj_test < 1:
            adj_train = len(features) - 1
            adj_test = 1
        if len(features) < 2:
             raise ValueError(f"Data tidak cukup untuk XGBoost pada {target_col}. Butuh minimal 2 baris data valid.")

        feat_train = features.iloc[:adj_train]
        feat_test = features.iloc[adj_train:]
        tgt_train = target.iloc[:adj_train]
        tgt_test = target.iloc[adj_train:]

        scaler = StandardScaler()
        feat_train_sc = scaler.fit_transform(feat_train)
        feat_test_sc = scaler.transform(feat_test)

        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            reg_alpha=0.1, reg_lambda=1.0,
        )
        xgb_model.fit(feat_train_sc, tgt_train)

        test_pred = pd.Series(xgb_model.predict(feat_test_sc), index=tgt_test.index)
        test_target = tgt_test

        fw = future_weather or {}
        forecast_vals_list = []
        temp_targets = list(df[target_col].values[-6:])
        temp_weather = {var: list(df[var].values[-6:]) for var in WEATHER_VARS}

        for step in range(forecast_months):
            new_feat = pd.DataFrame(index=[0])
            for var in WEATHER_VARS:
                current_val = fw.get(var, df[var].iloc[-1])
                new_feat[var] = current_val
                for i in range(1, 7):
                    tw = temp_weather[var]
                    new_feat[f"{var}_lag{i}"] = tw[-i] if i <= len(tw) else current_val
            for i in range(1, 7):
                new_feat[f"target_lag{i}"] = temp_targets[-i] if i <= len(temp_targets) else df[target_col].iloc[-1]
            for var in WEATHER_VARS:
                bl = best_lags.get(var, 0)
                if bl > 0:
                    tw = temp_weather[var]
                    new_feat[f"{var}_best_lag{bl}"] = tw[-bl] if bl <= len(tw) else fw.get(var, df[var].iloc[-1])
            month_num = (df["date"].iloc[-1].month + step) % 12 + 1
            new_feat["month"] = month_num
            new_feat["month_sin"] = np.sin(2 * np.pi * month_num / 12)
            new_feat["month_cos"] = np.cos(2 * np.pi * month_num / 12)

            new_feat = new_feat[feat_train.columns]
            new_feat_sc = scaler.transform(new_feat)
            pred_val = xgb_model.predict(new_feat_sc)[0]
            forecast_vals_list.append(pred_val)

            temp_targets.append(pred_val)
            for var in WEATHER_VARS:
                temp_weather[var].append(fw.get(var, df[var].iloc[-1]))

        forecast_vals = pd.Series(forecast_vals_list)
        residuals = test_pred.values - test_target.values
        std_resid = np.std(residuals) if len(residuals) > 1 else 0.1
        lower = forecast_vals - 1.96 * std_resid
        upper = forecast_vals + 1.96 * std_resid
        conf_int = pd.DataFrame({"lower": lower.values, "upper": upper.values})

        results["weather_lags"] = str(best_lags)
        results["models"] = "XGBoost"

    mae = mean_absolute_error(test_target, test_pred)
    rmse = np.sqrt(mean_squared_error(test_target, test_pred))
    r2 = r2_score(test_target, test_pred)
    results["mae"] = round(mae, 4)
    results["rmse"] = round(rmse, 4)
    results["r2"] = round(r2, 4)

    last_date = df["date"].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq="MS")

    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        f"forecast_{target_col}": forecast_vals.values,
        "lower_95": conf_int.iloc[:, 0].values,
        "upper_95": conf_int.iloc[:, 1].values,
    })

    return results, train_df, test_df, test_pred, forecast_df


site_names = ["Site B", "Site T", "Site L", "Site K", "Site S", "Site KL", "Site M"]
if "selected_site" not in st.session_state:
    st.session_state.selected_site = site_names[0]

selected_site = st.selectbox("Pilih Lokasi", site_names, index=site_names.index(st.session_state.selected_site), key="selected_site")
site_config = SITES[selected_site]
has_water_level = site_config.get("has_water_level", False)

st.title(f"Forecasting {selected_site}")
st.markdown(f"Analisis dan prediksi {'flowrate & water level' if has_water_level else 'flowrate'} berdasarkan data cuaca dari Open-Meteo (precipitation, evapotranspiration).")

if "added_rows" not in st.session_state or st.session_state.get("_last_site") != selected_site:
    st.session_state.added_rows = load_added_data(site_config["added_data_file"])
    if "forecast_results" in st.session_state:
        del st.session_state.forecast_results
    if "water_forecast_results" in st.session_state:
        del st.session_state.water_forecast_results
    st.session_state._last_site = selected_site

with st.spinner("Mengambil data cuaca dari Open-Meteo..."):
    try:
        df = get_data(site_config)
        weather_loaded = True
    except Exception as e:
        st.error(f"Gagal mengambil data cuaca dari Open-Meteo: {str(e)}. Menggunakan data rainfall dari file.")
        df = load_initial_data(site_config["data_file"])
        if "rainfall" in df.columns:
            df["precipitation"] = df["rainfall"]
        else:
            df["precipitation"] = 0.0
        df["et0"] = 4.0
        weather_loaded = False

TAB_OPTIONS = ["Data & Analisis", "Stationarity Check", "Model & Forecast", "Input Data Baru", "Export"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = TAB_OPTIONS[0]

active_tab = st.radio("Navigasi", TAB_OPTIONS, index=TAB_OPTIONS.index(st.session_state.active_tab), horizontal=True, key="active_tab", label_visibility="collapsed")

if active_tab == "Data & Analisis":
    st.header("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", f"{len(df)} bulan")
    col2.metric("Periode", f"{df['date'].dt.strftime('%Y-%m').iloc[0]} s/d {df['date'].dt.strftime('%Y-%m').iloc[-1]}")
    col3.metric("Flowrate Terakhir", f"{df['flowrate'].iloc[-1]:.2f}")

    if weather_loaded:
        st.success(f"Data cuaca Open-Meteo berhasil dimuat")

    st.subheader("Data Flowrate & Cuaca")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["date"], y=df["flowrate"], name="Flowrate", line=dict(color="#1f77b4", width=2)), secondary_y=False)
    if has_water_level and "water_level" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["water_level"], name="Water Level", line=dict(color="#2ca02c", width=2)), secondary_y=False)
    fig.add_trace(go.Bar(x=df["date"], y=df["precipitation"], name="Precipitation", marker_color="rgba(44,160,44,0.3)"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["date"], y=df["et0"], name="ET0 (mm)", line=dict(color="#9467bd", width=1.5, dash="dash")), secondary_y=False)
    fig.update_layout(title="Flowrate, Water Level, ET0 & Precipitation", height=500, hovermode="x unified")
    fig.update_yaxes(title_text="Flowrate / Water Level / ET0", secondary_y=False)
    fig.update_yaxes(title_text="Precipitation (mm)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    

    st.subheader("Lag Correlation: Variabel Cuaca → Flowrate")
    lag_results = {}
    for row_start in range(0, len(WEATHER_VARS), 3):
        row_vars = WEATHER_VARS[row_start:row_start + 3]
        cols_lag = st.columns(len(row_vars))
        for i, var in enumerate(row_vars):
            bl, corr_df = find_best_lag(df["flowrate"].values, df[var].values)
            lag_results[var] = {"best_lag": bl, "corr_df": corr_df}
            with cols_lag[i]:
                st.markdown(f"**{WEATHER_LABELS[var]}**")
                st.info(f"Best Lag: **{bl} bulan**")
                fig_corr = go.Figure()
                colors = ["#ff7f0e" if lag == bl else "#1f77b4" for lag in corr_df["Lag"]]
                fig_corr.add_trace(go.Bar(x=corr_df["Lag"], y=corr_df["Correlation"], marker_color=colors))
                fig_corr.update_layout(height=300, xaxis_title="Lag", yaxis_title="Corr", margin=dict(t=10))
                st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Data Tabel")
    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m")
    show_cols = ["date", "flowrate"]
    if has_water_level and "water_level" in display_df.columns:
        show_cols.append("water_level")
    if "rainfall" in display_df.columns:
        show_cols.append("rainfall")
    show_cols += WEATHER_VARS
    st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

elif active_tab == "Stationarity Check":
    st.header("Stationarity Check (ADF Test)")
    st.markdown("Augmented Dickey-Fuller test untuk mengecek apakah data stasioner (p-value < 0.05 = stasioner).")

    test_vars = [("Flowrate", df["flowrate"])] + [(WEATHER_LABELS[v], df[v]) for v in WEATHER_VARS]
    for row_start in range(0, len(test_vars), 4):
        row_vars = test_vars[row_start:row_start + 4]
        cols_adf = st.columns(len(row_vars))
        for i, (name, series) in enumerate(row_vars):
            adf_result = adf_test(series, name)
            with cols_adf[i]:
                st.subheader(name.split(" ")[0])
                status = "Stasioner" if adf_result["Stationary"] == "Yes" else "Tidak Stasioner"
                color = "green" if adf_result["Stationary"] == "Yes" else "red"
                st.markdown(f":{color}[**{status}**]")
                st.json(adf_result)

    if adf_test(df["flowrate"], "Flowrate")["Stationary"] == "No":
        st.subheader("Differencing Flowrate (1st order)")
        diff_flow = df["flowrate"].diff().dropna()
        adf_diff = adf_test(diff_flow, "Flowrate (1st diff)")
        status_d = "Stasioner" if adf_diff["Stationary"] == "Yes" else "Tidak Stasioner"
        color_d = "green" if adf_diff["Stationary"] == "Yes" else "red"
        st.markdown(f"Setelah differencing: :{color_d}[**{status_d}**]")
        st.json(adf_diff)

elif active_tab == "Model & Forecast":
    st.header("Model & Forecast")

    with st.form("model_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            model_choice = st.selectbox("Pilih Model", ["XGBoost", "ML (Ensemble RF+GBR)", "SARIMAX"])
        with col2:
            test_size = st.slider("Test Size (bulan)", min_value=6, max_value=min(36, len(df) // 3), value=6)
        with col3:
            forecast_months = st.slider("Forecast (bulan ke depan)", min_value=1, max_value=24, value=6)

        st.markdown("**Asumsi Cuaca untuk Forecast** (0 = pakai data terakhir)")
        cw1, cw2 = st.columns(2)
        with cw1:
            fw_precip = st.number_input("Precipitation (mm/bulan)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
        with cw2:
            fw_et0 = st.number_input("ET0 (mm/bulan)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

        run_model = st.form_submit_button("Jalankan Model", type="primary", use_container_width=True)

    # Run Model Logic
    if run_model:
        model_map = {
            "SARIMAX": "SARIMAX",
            "ML (Ensemble RF+GBR)": "ML",
            "XGBoost": "XGBoost",
        }
        model_key = model_map.get(model_choice, "XGBoost")
        future_weather_vals = {}
        if fw_precip > 0:
            future_weather_vals["precipitation"] = fw_precip
        if fw_et0 > 0:
            future_weather_vals["et0"] = fw_et0
        if not future_weather_vals:
            future_weather_vals = None
            
        with st.spinner("Sedang menjalankan model... Mohon tunggu."):
            try:
                # Flowrate forecast
                results, train_df, test_df, test_pred, forecast_df = run_forecast(
                    df, model_key, test_size, forecast_months, future_weather_vals, target_col="flowrate"
                )
                st.session_state.forecast_results = {
                    "results": results,
                    "train_df": train_df,
                    "test_df": test_df,
                    "test_pred": test_pred,
                    "forecast_df": forecast_df,
                    "model_choice": model_choice,
                }
                
                # Water level forecast if applicable
                if has_water_level:
                    w_results, w_train_df, w_test_df, w_test_pred, w_forecast_df = run_forecast(
                        df, model_key, test_size, forecast_months, future_weather_vals, target_col="water_level"
                    )
                    st.session_state.water_forecast_results = {
                        "results": w_results,
                        "train_df": w_train_df,
                        "test_df": w_test_df,
                        "test_pred": w_test_pred,
                        "forecast_df": w_forecast_df,
                        "model_choice": model_choice,
                    }
                st.success("Model berhasil dijalankan!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display Results Logic
    if "forecast_results" in st.session_state:
        fr = st.session_state.forecast_results
        res = fr["results"]
        fc_df = fr["forecast_df"]

        st.subheader("Flowrate Model Performance")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("MAE", f"{res['mae']:.4f}")
        mc2.metric("RMSE", f"{res['rmse']:.4f}")
        mc3.metric("R²", f"{res['r2']:.4f}")

        st.subheader("Flowrate Forecast Chart")
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=df["date"], y=df["flowrate"], name="Historical", line=dict(color="#1f77b4", width=2)))
        fig_fc.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast_flowrate"], name="Forecast", line=dict(color="#d62728", width=2)))
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
            y=pd.concat([fc_df["upper_95"], fc_df["lower_95"][::-1]]),
            fill="toself", fillcolor="rgba(214,39,40,0.15)", line=dict(color="rgba(255,255,255,0)"), name="95% CI",
        ))
        fig_fc.update_layout(title="Flowrate Forecast", xaxis_title="Date", yaxis_title="Flowrate", height=450)
        st.plotly_chart(fig_fc, use_container_width=True)
        st.subheader("Tabel Forecast Flowrate (Preview)")
        cols = ["date", "forecast_flowrate", "lower_95", "upper_95"]
        preview_fc = fc_df[cols].copy()
        preview_fc["date"] = preview_fc["date"].dt.strftime("%Y-%m")
        st.dataframe(preview_fc, use_container_width=True, hide_index=True)
        
        if has_water_level and "water_forecast_results" in st.session_state:
            wr = st.session_state.water_forecast_results
            w_res = wr["results"]
            w_fc_df = wr["forecast_df"]
            
            st.subheader("Water Level Model Performance")
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("MAE", f"{w_res['mae']:.4f}")
            wc2.metric("RMSE", f"{w_res['rmse']:.4f}")
            wc3.metric("R²", f"{w_res['r2']:.4f}")

            st.subheader("Water Level Forecast Chart")
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(x=df["date"], y=df["water_level"], name="Historical", line=dict(color="#1f77b4", width=2)))
            fig_w.add_trace(go.Scatter(x=w_fc_df["date"], y=w_fc_df["forecast_water_level"], name="Forecast", line=dict(color="#2ca02c", width=2)))
            fig_w.add_trace(go.Scatter(
                x=pd.concat([w_fc_df["date"], w_fc_df["date"][::-1]]),
                y=pd.concat([w_fc_df["upper_95"], w_fc_df["lower_95"][::-1]]),
                fill="toself", fillcolor="rgba(44,160,44,0.15)", line=dict(color="rgba(255,255,255,0)"), name="95% CI",
            ))
            fig_w.update_layout(title="Water Level Forecast", xaxis_title="Date", yaxis_title="Water Level", height=450)
            st.plotly_chart(fig_w, use_container_width=True)
            st.subheader("Tabel Forecast Water Level (Preview)")
            cols_w = ["date", "forecast_water_level", "lower_95", "upper_95"]
            preview_w = w_fc_df[cols_w].copy()
            preview_w["date"] = preview_w["date"].dt.strftime("%Y-%m")
            st.dataframe(preview_w, use_container_width=True, hide_index=True)


elif active_tab == "Input Data Baru":
    st.header("Input Data Baru")
    st.markdown("Tambahkan beberapa data sekaligus. Data cuaca otomatis diambil dari Open-Meteo.")

    last_date = df["date"].iloc[-1]
    next_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq="MS")

    with st.form("input_data_form"):
        st.markdown("**Isi flowrate di bawah** (data cuaca akan otomatis diambil dari Open-Meteo):")
        input_dates = []
        input_flowrates = []
        input_waterlevels = []
        
        for i in range(6):
            if has_water_level:
                c1, c2, c3 = st.columns([1, 1, 1])
            else:
                c1, c2 = st.columns([1, 1])

            with c1:
                d = st.text_input(
                    f"Date {i+1} (YYYY-MM)",
                    value=next_dates[i].strftime("%Y-%m"),
                    key=f"date_{i}"
                )

            with c2:
                f = st.number_input(
                    f"Flowrate {i+1}",
                    min_value=0.0, max_value=1000.0,
                    value=0.0, step=0.01,
                    key=f"flow_{i}"
                )

            input_dates.append(d)
            input_flowrates.append(f)

            if has_water_level:
                with c3:
                    wl = st.number_input(
                        f"Water Level {i+1}",
                        min_value=0.0, max_value=100.0,
                        value=0.0, step=0.01,
                        key=f"wl_{i}"
                    )
                input_waterlevels.append(wl)


        submitted_data = st.form_submit_button("Tambah Semua Data", type="primary", use_container_width=True)

    if submitted_data:
        edited_df = pd.DataFrame({
            "date": input_dates,
            "flowrate": input_flowrates
        })

        if has_water_level:
            edited_df["water_level"] = input_waterlevels
        valid_rows = edited_df[edited_df["flowrate"] > 0].copy()
        if has_water_level:
            valid_rows = valid_rows[valid_rows["water_level"] > 0].copy()
        if len(valid_rows) == 0:
            st.error("Tidak ada data valid (flowrate masih 0 semua).")
        else:
            existing_dates = df["date"].tolist()
            added_count = 0
            skipped = []
            new_rows_to_add = []
            for _, row in valid_rows.iterrows():
                try:
                    date_dt = pd.to_datetime(row["date"], format="%Y-%m")
                    if date_dt in existing_dates:
                        skipped.append(row["date"])
                    else:
                        new_row = {
                            "date": row["date"],
                            "flowrate": float(row["flowrate"]),
                        }

                        if has_water_level and "water_level" in row:
                            new_row["water_level"] = float(row["water_level"])

                        new_rows_to_add.append(new_row)
                        existing_dates.append(date_dt)
                        added_count += 1
                except Exception:
                    skipped.append(str(row["date"]))

            if added_count > 0:
                weather_fetched = False
                try:
                    dates_to_fetch = [pd.to_datetime(r["date"], format="%Y-%m") for r in new_rows_to_add]
                    min_date = min(dates_to_fetch).strftime("%Y-%m-%d")
                    max_date = (max(dates_to_fetch) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    weather_new = fetch_openmeteo_monthly(min_date, max_date, site_config["latitude"], site_config["longitude"])
                    weather_dict = {}
                    for _, wr in weather_new.iterrows():
                        key = wr["date"].strftime("%Y-%m")
                        w_entry = {"rainfall": wr["precipitation"]}
                        for wv in WEATHER_VARS:
                            w_entry[wv] = wr[wv]
                        weather_dict[key] = w_entry
                    for nr in new_rows_to_add:
                        if nr["date"] in weather_dict:
                            nr.update(weather_dict[nr["date"]])
                            weather_fetched = True
                        else:
                            for wv in WEATHER_VARS:
                                nr[wv] = df[wv].iloc[-1]
                            nr["rainfall"] = df["precipitation"].iloc[-1]
                except Exception:
                    for nr in new_rows_to_add:
                        for wv in WEATHER_VARS:
                            nr[wv] = df[wv].iloc[-1]
                        nr["rainfall"] = df["precipitation"].iloc[-1]
                    st.warning("Gagal mengambil data cuaca dari Open-Meteo. Menggunakan data cuaca terakhir sebagai pengganti.")

                for nr in new_rows_to_add:
                    st.session_state.added_rows.append(nr)
                save_added_data(st.session_state.added_rows, site_config["added_data_file"])

                if "forecast_results" in st.session_state:
                    del st.session_state.forecast_results
                load_data_with_weather.clear()

                if weather_fetched:
                    st.success(f"**{added_count} data** berhasil ditambahkan dengan data cuaca dari Open-Meteo!")
                else:
                    st.success(f"**{added_count} data** berhasil ditambahkan.")
            if skipped:
                st.warning(f"Data berikut dilewati (sudah ada/format salah): {', '.join(skipped)}")
            if added_count > 0:
                st.rerun()

    if len(st.session_state.added_rows) > 0:
        st.subheader("Data yang Sudah Ditambahkan")
        added_display = pd.DataFrame(st.session_state.added_rows)
        st.dataframe(added_display, use_container_width=True, hide_index=True)
        if st.button("Reset Data Tambahan", type="secondary"):
            st.session_state.added_rows = []
            save_added_data([], site_config["added_data_file"])
            if "forecast_results" in st.session_state:
                del st.session_state.forecast_results
            if "water_forecast_results" in st.session_state:
                del st.session_state.water_forecast_results
            load_data_with_weather.clear()
            st.rerun()

elif active_tab == "Export":
    st.header("Export Hasil")

    if "forecast_results" in st.session_state:
        forecast_df = st.session_state.forecast_results["forecast_df"]

        st.subheader("Export Forecast ke CSV")
        export_df = forecast_df.copy()
        export_df["date"] = export_df["date"].dt.strftime("%Y-%m")
        csv = export_df.to_csv(index=False)
        st.download_button(label="Download Forecast CSV", data=csv, file_name="forecast_flowrate.csv", mime="text/csv", type="primary", use_container_width=True)

        st.subheader("Preview Export")
        st.dataframe(export_df, use_container_width=True, hide_index=True)

        st.subheader("Export Data Lengkap + Forecast")
        full_export = df.copy()
        full_export["date"] = full_export["date"].dt.strftime("%Y-%m")
        full_export["type"] = "historical"

        fc_export = forecast_df.copy()
        fc_export["date"] = fc_export["date"].dt.strftime("%Y-%m")
        fc_export = fc_export.rename(columns={"forecast_flowrate": "flowrate"})
        fc_export["type"] = "forecast"

        base_cols = ["date", "flowrate", "type"]
        combined = pd.concat([full_export[base_cols], fc_export[base_cols]], ignore_index=True)
        csv_full = combined.to_csv(index=False)
        st.download_button(label="Download Data Lengkap + Forecast CSV", data=csv_full, file_name="full_data_with_forecast.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("Jalankan model terlebih dahulu di tab 'Model & Forecast' untuk mengekspor hasil.")
