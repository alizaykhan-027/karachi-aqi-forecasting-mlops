# =============================================================================
# AQI FEATURE ENGINEERING PIPELINE
# SAFE TIME-SERIES FEATURE STORE PIPELINE
# NO DATA LEAKAGE
# READY FOR SUPABASE FEATURE STORE
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

import os
import numpy as np
import pandas as pd

# =============================================================================
# CREATE PROCESSED DIRECTORY
# =============================================================================

os.makedirs("data/processed", exist_ok=True)

# =============================================================================
# LOAD RAW DATA
# =============================================================================

raw_df = pd.read_csv(
    "data/raw/karachi_aqi_historical.csv"
)

print("=" * 70)
print("RAW DATA LOADED")
print("=" * 70)

print("\nRaw Shape:")
print(raw_df.shape)

# =============================================================================
# FEATURE ENGINEERING FUNCTION
# =============================================================================

def engineer_aqi_features(df):

    # =========================================================================
    # COPY DATA
    # =========================================================================

    df = df.copy()

    # =========================================================================
    # SORT BY TIME
    # =========================================================================

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # =========================================================================
    # REMOVE ESSENTIAL NULLS
    # =========================================================================

    essential_cols = ["pm2_5", "us_aqi"]

    df = df.dropna(subset=essential_cols)

    # =========================================================================
    # BASIC FEATURES
    # =========================================================================

    df["pm25_log"] = np.log1p(df["pm2_5"])

    # =========================================================================
    # TIME FEATURES
    # =========================================================================

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month_of_year"] = df["timestamp"].dt.month
    df["day_of_month"] = df["timestamp"].dt.day

    df["week_of_year"] = (
        df["timestamp"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # =========================================================================
    # SEASON FEATURE
    # =========================================================================

    def get_season(month):

        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3

    df["season"] = (
        df["month_of_year"]
        .apply(get_season)
    )

    # =========================================================================
    # CYCLICAL FEATURES
    # =========================================================================

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour_of_day"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour_of_day"] / 24
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month_of_year"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month_of_year"] / 12
    )

    # =========================================================================
    # WIND FEATURES
    # =========================================================================

    if "wind_direction_10m" in df.columns:

        radians = np.deg2rad(
            df["wind_direction_10m"]
        )

        df["wind_x"] = np.cos(radians)
        df["wind_y"] = np.sin(radians)

    # =========================================================================
    # AQI LAG FEATURES
    # =========================================================================

    df["aqi_lag_1h"] = df["us_aqi"].shift(1)
    df["aqi_lag_6h"] = df["us_aqi"].shift(6)
    df["aqi_lag_24h"] = df["us_aqi"].shift(24)

    # =========================================================================
    # AQI ROLLING FEATURES
    # =========================================================================

    df["aqi_rolling_avg_6h"] = (
        df["us_aqi"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    df["aqi_rolling_avg_24h"] = (
        df["us_aqi"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    # =========================================================================
    # AQI CHANGE RATE
    # =========================================================================

    df["aqi_change_rate_24h"] = (
        df["us_aqi"]
        .diff(24)
    )

    # =========================================================================
    # PM2.5 LAG FEATURES
    # =========================================================================

    lag_windows = [1, 2, 3, 6, 12, 24, 48]

    for lag in lag_windows:
        df[f"pm25_lag_{lag}"] = (
            df["pm2_5"]
            .shift(lag)
        )

    # =========================================================================
    # PM10 FEATURES
    # =========================================================================

    if "pm10" in df.columns:

        for lag in [6, 24]:
            df[f"pm10_lag_{lag}"] = (
                df["pm10"]
                .shift(lag)
            )

        df["pm_ratio"] = (
            df["pm2_5"] /
            (df["pm10"] + 1e-6)
        )

    # =========================================================================
    # ROLLING FEATURES
    # =========================================================================

    rolling_windows = [3, 6, 12, 24]

    for w in rolling_windows:

        df[f"pm25_roll_mean_{w}"] = (
            df["pm2_5"]
            .shift(1)
            .rolling(w)
            .mean()
        )

    for w in [6, 12, 24]:

        df[f"pm25_roll_std_{w}"] = (
            df["pm2_5"]
            .shift(1)
            .rolling(w)
            .std()
        )

    for w in [6, 24]:

        df[f"pm25_roll_min_{w}"] = (
            df["pm2_5"]
            .shift(1)
            .rolling(w)
            .min()
        )

    for w in [3, 6, 24]:

        df[f"pm25_roll_max_{w}"] = (
            df["pm2_5"]
            .shift(1)
            .rolling(w)
            .max()
        )

    df["pm25_median_24"] = (
        df["pm2_5"]
        .shift(1)
        .rolling(24)
        .median()
    )

    # =========================================================================
    # EMA FEATURES
    # =========================================================================

    for span in [3, 6, 12, 24]:

        df[f"pm25_ema_{span}"] = (
            df["pm2_5"]
            .shift(1)
            .ewm(span=span, adjust=False)
            .mean()
        )

    df["pm25_ewm_std_24"] = (
        df["pm2_5"]
        .shift(1)
        .ewm(span=24)
        .std()
    )

    df["pm25_volatility_24"] = (
        df["pm2_5"]
        .shift(1)
        .rolling(24)
        .std()
    )

    df["pm25_cumulative_24"] = (
        df["pm2_5"]
        .shift(1)
        .rolling(24)
        .sum()
    )

    # =========================================================================
    # POLLUTION FLAGS
    # =========================================================================

    df["pollution_spike"] = (
        df["pm2_5"] >
        (
            df["pm2_5"]
            .shift(1)
            .rolling(24)
            .mean()
            * 1.5
        )
    ).astype(int)

    df["high_pollution_flag"] = (
        df["pm2_5"] > 35
    ).astype(int)

    # =========================================================================
    # WEATHER FEATURES
    # =========================================================================

    weather_cols = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ]

    for col in weather_cols:

        if col in df.columns:

            df[f"{col}_lag_6"] = (
                df[col]
                .shift(6)
            )

            df[f"{col}_roll_mean_24"] = (
                df[col]
                .shift(1)
                .rolling(24)
                .mean()
            )

    # =========================================================================
    # TARGETS (FOR TRAINING ONLY)
    # =========================================================================

    df["target_24h"] = (
        df["pm2_5"]
        .shift(-24)
    )

    df["target_48h"] = (
        df["pm2_5"]
        .shift(-48)
    )

    df["target_72h"] = (
        df["pm2_5"]
        .shift(-72)
    )

    # =========================================================================
    # CLEANUP
    # =========================================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Do NOT drop based on target columns
    feature_cols = [
        col for col in df.columns
        if not col.startswith("target_")
    ]

    df = (
        df
        .dropna(subset=feature_cols)
        .reset_index(drop=True)
    )

    return df


# =============================================================================
# APPLY FEATURE ENGINEERING
# =============================================================================

df_features = engineer_aqi_features(raw_df)

# =============================================================================
# KEEP ONLY LATEST ROW FOR HOURLY PIPELINE
# =============================================================================

df_features = (
    df_features
    .sort_values("timestamp")
    .tail(1)
    .reset_index(drop=True)
)

# =============================================================================
# INFORMATION
# =============================================================================

print("=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print("\nFinal Shape:")
print(df_features.shape)

print("\nTotal Features:")
print(len(df_features.columns))

print("\nColumns:")
print(df_features.columns.tolist())

# =============================================================================
# SAVE ENGINEERED FEATURES
# =============================================================================

df_features.to_csv(
    "data/processed/aqi_feature_store.csv",
    index=False
)

print("\n✅ Feature store CSV saved successfully.")
print("File: data/processed/aqi_feature_store.csv")
