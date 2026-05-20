# =========================================================
# AQI MODEL TRAINING PIPELINE (PRODUCTION VERSION)
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

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials")

# =========================================================
# SUPABASE CONNECTION
# =========================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

print("✅ Supabase connected")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_previous_performance(horizon):
    response = (
        supabase
        .table("model_performance")
        .select("*")
        .eq("horizon", horizon)
        .eq("is_production", True)
        .execute()
    )

    if response.data:
        return response.data[0]

    return None


def save_model_version(
    version_name,
    horizon,
    model_name,
    mae,
    rmse,
    r2,
    promoted
):
    supabase.table("model_versions").insert({
        "version_name": version_name,
        "horizon": horizon,
        "model_name": model_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "promoted": promoted
    }).execute()


# =========================================================
# FETCH DATA FROM SUPABASE
# =========================================================

print("\nFetching AQI feature dataset...")

all_data = []
batch_size = 1000
start = 0

while True:
    response = (
        supabase
        .table("aqi_features")
        .select("*")
        .range(start, start + batch_size - 1)
        .execute()
    )

    batch = response.data

    if not batch:
        break

    all_data.extend(batch)

    print(f"Collected {len(all_data)} rows")

    if len(batch) < batch_size:
        break

    start += batch_size

if not all_data:
    raise ValueError("No data returned from Supabase")

print(f"\n✅ Total rows fetched: {len(all_data)}")

# =========================================================
# DATAFRAME PREP
# =========================================================

df_raw = pd.DataFrame(all_data)

df_raw["timestamp"] = pd.to_datetime(
    df_raw["timestamp"],
    errors="coerce"
)

df_raw = df_raw.dropna(
    subset=["timestamp"]
).reset_index(drop=True)

df_raw = (
    df_raw
    .sort_values("timestamp")
    .drop_duplicates(subset=["timestamp"])
    .reset_index(drop=True)
)

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
    "pm2_5",
    "us_aqi",
    "pm25_log",

    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",

    "is_weekend",
    "season",

    "aqi_lag_1h",
    "aqi_lag_6h",
    "aqi_lag_24h",
    "aqi_rolling_avg_6h",
    "aqi_rolling_avg_24h",
    "aqi_change_rate_24h",

    "pm25_lag_1",
    "pm25_lag_2",
    "pm25_lag_3",
    "pm25_lag_6",
    "pm25_lag_12",
    "pm25_lag_24",
    "pm25_lag_48",

    "pm25_roll_mean_3",
    "pm25_roll_mean_6",
    "pm25_roll_mean_12",
    "pm25_roll_mean_24",

    "pm25_roll_std_24",
    "pm25_roll_min_24",
    "pm25_roll_max_24",
    "pm25_median_24",

    "pm25_ema_3",
    "pm25_ema_6",
    "pm25_ema_12",
    "pm25_ema_24",

    "pm25_volatility_24",
    "pm25_cumulative_24",

    "high_pollution_flag"
]

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

priority_features = [
    col for col in priority_features
    if col in df_final.columns
]

print("\nTotal Features:", len(priority_features))

targets = [
    "target_24h",
    "target_48h",
    "target_72h"
]

all_needed = priority_features + targets

df_final = df_final.dropna(
    subset=all_needed
).reset_index(drop=True)

if len(df_final) < 500:
    raise ValueError("Dataset too small after cleaning")

print("\nFinal Dataset Shape:", df_final.shape)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X = df_final[priority_features]
y = df_final[targets]

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

horizons = ["24h", "48h", "72h"]

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
# SAVE / COMPARE MODELS
# =========================================================

os.makedirs("models", exist_ok=True)

version_name = pd.Timestamp.now().strftime("aqi_model_%Y%m%d_%H%M%S")

for horizon_index, horizon_name in enumerate(horizons):

    horizon_results = results_df[
        results_df["Horizon"] == horizon_name
    ]

    best_row = horizon_results.loc[
        horizon_results["RMSE"].idxmin()
    ]

    new_model_name = best_row["Model"]
    new_mae = best_row["MAE"]
    new_rmse = best_row["RMSE"]
    new_r2 = best_row["R2"]

    previous = get_previous_performance(horizon_name)

    promote = False

    if previous is None:
        print(f"\nNo previous production model for {horizon_name}")
        promote = True

    elif new_rmse < previous["rmse"]:
        print(f"\nNew model improved for {horizon_name}")
        promote = True

    else:
        print(f"\nOld production model remains for {horizon_name}")
        promote = False

    save_model_version(
        version_name=version_name,
        horizon=horizon_name,
        model_name=new_model_name,
        mae=new_mae,
        rmse=new_rmse,
        r2=new_r2,
        promoted=promote
    )

    if promote:

        best_model = trained_models[
            new_model_name
        ][horizon_index]

        save_path = f"models/best_model_{horizon_name}.pkl"

        joblib.dump(
            best_model,
            save_path
        )

        supabase.table("model_performance").update({
            "is_production": False
        }).eq("horizon", horizon_name).execute()

        supabase.table("model_performance").insert({
            "horizon": horizon_name,
            "model_name": new_model_name,
            "mae": float(new_mae),
            "rmse": float(new_rmse),
            "r2": float(new_r2),
            "is_production": True
        }).execute()

        print(f"✅ Promoted new model for {horizon_name}")

    else:
        print(f"⏭ Kept old production model for {horizon_name}")

print("\n✅ Daily AQI training pipeline completed successfully")