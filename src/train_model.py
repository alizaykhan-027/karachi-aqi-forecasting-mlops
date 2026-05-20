# =========================================================
# AQI MODEL TRAINING PIPELINE
# =========================================================

import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone

from dotenv import load_dotenv
from supabase import create_client

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# =========================================================
# SUPABASE CONNECTION
# =========================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

print("✅ Supabase connected")

# =========================================================
# FETCH DATA FROM SUPABASE
# =========================================================

print("\nFetching AQI feature dataset...")

# No ORDER BY (avoids timeout)
response = (
    supabase
    .table("aqi_features")
    .select("*")
    .limit(10000)
    .execute()
)

all_data = response.data

if not all_data:
    raise ValueError("No data returned from Supabase")

print(f"Collected {len(all_data)} rows")

# =========================================================
# DATAFRAME
# =========================================================

df_raw = pd.DataFrame(all_data)

# =========================================================
# DATETIME
# =========================================================

df_raw["timestamp"] = pd.to_datetime(
    df_raw["timestamp"],
    errors="coerce"
)

# Remove bad timestamps
df_raw = df_raw.dropna(
    subset=["timestamp"]
).reset_index(drop=True)

# =========================================================
# SORT
# =========================================================

df_raw = (
    df_raw
    .sort_values("timestamp")
    .drop_duplicates(subset=["timestamp"])
    .reset_index(drop=True)
)

# =========================================================
# REMOVE INVALID VALUES
# =========================================================

df_raw = df_raw.replace(
    [np.inf, -np.inf],
    np.nan
)

print(f"\n✅ Loaded {len(df_raw)} rows")

# =========================================================
# REQUIRED TARGETS
# =========================================================

required_targets = [
    "target_24h",
    "target_48h",
    "target_72h"
]

df_final = df_raw.dropna(
    subset=required_targets
).reset_index(drop=True)

# =========================================================
# FEATURES
# =========================================================

priority_features = [

    # BASIC
    "pm2_5",
    "us_aqi",
    "pm25_log",

    # TIME
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",

    # FLAGS
    "is_weekend",
    "season",

    # AQI FEATURES
    "aqi_lag_1h",
    "aqi_lag_6h",
    "aqi_lag_24h",
    "aqi_rolling_avg_6h",
    "aqi_rolling_avg_24h",
    "aqi_change_rate_24h",

    # PM2.5 LAGS
    "pm25_lag_1",
    "pm25_lag_2",
    "pm25_lag_3",
    "pm25_lag_6",
    "pm25_lag_12",
    "pm25_lag_24",
    "pm25_lag_48",

    # ROLLING
    "pm25_roll_mean_3",
    "pm25_roll_mean_6",
    "pm25_roll_mean_12",
    "pm25_roll_mean_24",

    # STD
    "pm25_roll_std_24",

    # MIN/MAX
    "pm25_roll_min_24",
    "pm25_roll_max_24",

    # MEDIAN
    "pm25_median_24",

    # EMA
    "pm25_ema_3",
    "pm25_ema_6",
    "pm25_ema_12",
    "pm25_ema_24",

    # VOLATILITY
    "pm25_volatility_24",

    # CUMULATIVE
    "pm25_cumulative_24",

    # FLAGS
    "high_pollution_flag"
]

# =========================================================
# OPTIONAL FEATURES
# =========================================================

optional_cols = [
    "pm10",
    "pm_ratio",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure"
]

for col in optional_cols:
    if col in df_final.columns:
        priority_features.append(col)

# =========================================================
# KEEP AVAILABLE FEATURES ONLY
# =========================================================

priority_features = [
    col for col in priority_features
    if col in df_final.columns
]

print("\nTotal Features:", len(priority_features))

# =========================================================
# TARGETS
# =========================================================

targets = [
    "target_24h",
    "target_48h",
    "target_72h"
]

# =========================================================
# FINAL CLEANING
# =========================================================

all_needed = priority_features + targets

df_final = df_final.dropna(
    subset=all_needed
).reset_index(drop=True)

if len(df_final) < 500:
    raise ValueError("Dataset too small after cleaning")

print("\nFinal Dataset Shape:", df_final.shape)

# =========================================================
# INPUTS / TARGETS
# =========================================================

X = df_final[priority_features]
y = df_final[targets]

# =========================================================
# TIME SERIES SPLIT
# =========================================================

n = len(df_final)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_test = y.iloc[val_end:]

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =========================================================
# MODEL CONFIGS
# =========================================================

model_configs = {

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
}

# =========================================================
# TRAIN MODELS
# =========================================================

results = []
trained_models = {}

for model_name, base_model in model_configs.items():

    print(f"\n================ {model_name.upper()} =================")

    predictions = []
    horizon_models = []

    for i in range(3):

        sample_weights = np.where(
            y_train.iloc[:, i] < 35,
            2.5,
            1.0
        )

        model = clone(base_model)

        model.fit(
            X_train,
            y_train.iloc[:, i],
            sample_weight=sample_weights
        )

        preds = model.predict(X_test)

        predictions.append(preds)
        horizon_models.append(model)

    predictions = np.column_stack(predictions)
    trained_models[model_name] = horizon_models

    horizons = ["24h", "48h", "72h"]

    for i, h in enumerate(horizons):

        mae = mean_absolute_error(
            y_test.iloc[:, i],
            predictions[:, i]
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test.iloc[:, i],
                predictions[:, i]
            )
        )

        r2 = r2_score(
            y_test.iloc[:, i],
            predictions[:, i]
        )

        results.append({
            "Model": model_name,
            "Horizon": h,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 3)
        })

        print(f"\n{h} Forecast")
        print(f"MAE  : {mae:.3f}")
        print(f"RMSE : {rmse:.3f}")
        print(f"R2   : {r2:.3f}")

# =========================================================
# RESULTS
# =========================================================

results_df = pd.DataFrame(results)

print("\n================ FINAL RESULTS ================\n")
print(results_df)

# =========================================================
# SAVE BEST MODELS
# =========================================================

os.makedirs("models", exist_ok=True)

horizons = ["24h", "48h", "72h"]

for horizon_index, horizon_name in enumerate(horizons):

    horizon_results = results_df[
        results_df["Horizon"] == horizon_name
    ]

    best_row = horizon_results.loc[
        horizon_results["RMSE"].idxmin()
    ]

    best_model_name = best_row["Model"]

    best_model = trained_models[
        best_model_name
    ][horizon_index]

    save_path = f"models/best_model_{horizon_name}.pkl"

    joblib.dump(
        best_model,
        save_path
    )

    print(
        f"\n✅ Saved Best {horizon_name} Model: "
        f"{best_model_name}"
    )

print("\n✅ All best models saved successfully")