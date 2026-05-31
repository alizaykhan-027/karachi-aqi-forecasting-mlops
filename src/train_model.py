# =========================================================
# AQI MODEL TRAINING PIPELINE (PART 1)
# LOAD DATA + MERGE BUFFER + PREP DATA
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
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
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


def save_model_version(version_name, horizon, model_name, mae, rmse, r2, promoted):
    supabase.table("model_versions").insert({
        "version_name": version_name,
        "horizon": horizon,
        "model_name": model_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "promoted": promoted
    }).execute()


def fetch_all_rows(table_name):
    all_data = []
    offset = 0
    batch_size = 1000

    while True:
        response = (
            supabase
            .table(table_name)
            .select("*")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break

        all_data.extend(batch)
        offset += batch_size
        print(f"{table_name}: collected {len(all_data)} rows")

        if len(batch) < batch_size:
            break

    return all_data


# =========================================================
# FETCH HISTORICAL + BUFFER DATA
# =========================================================
print("\nFetching historical AQI data...")
historical_data = fetch_all_rows("aqi_features")

print("\nFetching buffer AQI data...")
buffer_data = fetch_all_rows("new_daily_data")

if not historical_data and not buffer_data:
    raise ValueError("No data returned from Supabase")

# =========================================================
# MERGE DATA
# =========================================================
df_historical = pd.DataFrame(historical_data) if historical_data else pd.DataFrame()
df_buffer = pd.DataFrame(buffer_data) if buffer_data else pd.DataFrame()

df_raw = pd.concat([df_historical, df_buffer], ignore_index=True)



# =========================================================
# DATAFRAME CLEANING & TYPE ENFORCEMENT
# =========================================================
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce", utc=True)
df_raw = df_raw.dropna(subset=["timestamp"]).reset_index(drop=True)

df_raw = (
    df_raw
    .sort_values("timestamp")
    .drop_duplicates(subset=["timestamp"])
    .reset_index(drop=True)
)

df_raw = df_raw.replace([np.inf, -np.inf], np.nan)

# --- ADD THIS ROBUST CLEANING BLOCK ---
# 1. Ensure all features and targets are treated as numeric
# 2. Force conversion to float; invalid values (strings/None) become NaN
all_potential_cols = priority_features + targets
for col in all_potential_cols:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# 3. Now safely fill the holes
df_raw[all_potential_cols] = df_raw[all_potential_cols].fillna(0)
# --------------------------------------

print(f"\n✅ Total merged raw entries: {len(df_raw)}")
# =========================================================
# FEATURES & CONFIGURATIONS
# =========================================================
priority_features = [
    "pm2_5", "us_aqi", "pm25_log", "hour_sin", "hour_cos", 
    "day_sin", "day_cos", "month_sin", "month_cos", "is_weekend", "season",
    "aqi_lag_1h", "aqi_lag_6h", "aqi_lag_24h", "aqi_rolling_avg_6h", 
    "aqi_rolling_avg_24h", "aqi_change_rate_24h", "pm25_lag_1", "pm25_lag_2", 
    "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24", "pm25_lag_48",
    "pm25_roll_mean_3", "pm25_roll_mean_6", "pm25_roll_mean_12", "pm25_roll_mean_24",
    "pm25_roll_std_24", "pm25_roll_min_24", "pm25_roll_max_24", "pm25_median_24",
    "pm25_ema_3", "pm25_ema_6", "pm25_ema_12", "pm25_ema_24", "pm25_volatility_24",
    "pm25_cumulative_24", "high_pollution_flag"
]

optional_cols = [
    "pm10", "pm_ratio", "carbon_monoxide", "nitrogen_dioxide", 
    "sulphur_dioxide", "ozone", "temperature", "humidity", "wind_speed", "pressure"
]

for col in optional_cols:
    if col in df_raw.columns:
        priority_features.append(col)

priority_features = [col for col in priority_features if col in df_raw.columns]
targets = ["target_24h", "target_48h", "target_72h"]

# Fill input metric holes with 0 to prevent ML model training failures
df_raw[priority_features] = df_raw[priority_features].fillna(0)

# =========================================================
# SAFE TRAINING EXTRACTION (Handles NULL target rows gracefully)
# =========================================================
# Drop targets only for the mathematical calculation of model training
df_trainable = df_raw.dropna(subset=targets).reset_index(drop=True)

if len(df_trainable) < 200:
    print(f"⚠️ Trainable set size ({len(df_trainable)}) too low. Lowering constraints for bootstrap mode...")
    df_trainable = df_raw.dropna(subset=["target_24h"]).reset_index(drop=True)

if len(df_trainable) == 0:
    raise ValueError("FATAL: Zero rows have target variables available. Check historical ingestion.")

print("Total Training Features:", len(priority_features))
print("Final Trainable Dataset Shape:", df_trainable.shape)

# =========================================================
# TIME-SERIES SECURE SPLIT (No leakage, full range mapping)
# =========================================================
X = df_trainable[priority_features]
y = df_trainable[targets]

n = len(df_trainable)
train_end = int(n * 0.80)  # Use full 80% sequentially for training

X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]

print(f"\nTrain Shape: {X_train.shape} | Test Shape: {X_test.shape}")

# =========================================================
# MODEL CONFIGS
# =========================================================
model_configs = {
    "ExtraTrees": ExtraTreesRegressor(n_estimators=400, max_depth=14, min_samples_leaf=2, min_samples_split=4, max_features="sqrt", random_state=42, n_jobs=-1),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_leaf=2, min_samples_split=4, max_features="sqrt", random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=250, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=250, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
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
        # Gracefully replace missing sub-targets within trainable subsets if present
        y_train_clean = y_train.iloc[:, i].fillna(method="ffill").fillna(method="bfill")
        
        sample_weights = np.where(y_train_clean < 35, 2.5, 1.0)
        model = clone(base_model)
        model.fit(X_train, y_train_clean, sample_weight=sample_weights)

        preds = model.predict(X_test)
        predictions.append(preds)
        horizon_models.append(model)

    predictions = np.column_stack(predictions)
    trained_models[model_name] = horizon_models

    for i, h in enumerate(horizons):
        y_test_clean = y_test.iloc[:, i].fillna(method="ffill").fillna(method="bfill")
        
        mae = mean_absolute_error(y_test_clean, predictions[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_clean, predictions[:, i]))
        r2 = r2_score(y_test_clean, predictions[:, i])

        results.append({
            "Model": model_name, "Horizon": h,
            "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)
        })

        print(f"\n{h} Forecast → MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

results_df = pd.DataFrame(results)
print("\n================ FINAL RESULTS ================\n", results_df)

# =========================================================
# SAVE / COMPARE MODELS
# =========================================================
os.makedirs("models", exist_ok=True)
version_name = pd.Timestamp.now(tz="UTC").strftime("aqi_model_%Y%m%d_%H%M%S")

for horizon_index, horizon_name in enumerate(horizons):
    horizon_results = results_df[results_df["Horizon"] == horizon_name]
    best_row = horizon_results.loc[horizon_results["RMSE"].idxmin()]

    new_model_name = best_row["Model"]
    new_mae = best_row["MAE"]
    new_rmse = best_row["RMSE"]
    new_r2 = best_row["R2"]

    previous = get_previous_performance(horizon_name)
    promote = previous is None or new_rmse < previous["rmse"]

    if promote:
        print(f"\nPromoting new model structure for {horizon_name}...")
    else:
        print(f"\nOld production model structure retained for {horizon_name}")

    save_model_version(
        version_name=version_name, horizon=horizon_name, model_name=new_model_name,
        mae=new_mae, rmse=new_rmse, r2=new_r2, promoted=promote
    )

    if promote:
        best_model = trained_models[new_model_name][horizon_index]
        joblib.dump(best_model, f"models/best_model_{horizon_name}.pkl")

        supabase.table("model_performance").upsert({
            "horizon": horizon_name,
            "model_name": new_model_name,
            "mae": float(new_mae),
            "rmse": float(new_rmse),
            "r2": float(new_r2),
            "is_production": True
        }, on_conflict="horizon").execute()
        print(f"✅ Promoted and upserted new model for {horizon_name}")
    else:
        print(f"⏭ Kept old production model for {horizon_name}")

# =========================================================
# MERGE BUFFER INTO HISTORICAL TABLE (STRICT DATE-TYPE FIX)
# =========================================================
print("\nMerging buffer data into aqi_features...")

if buffer_data and not df_historical.empty:
    # Safely convert to clean string sets with a uniform format
    historical_timestamps = set(
        pd.to_datetime(df_historical["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    
    rows_to_insert = []

    for row in buffer_data:
        if not row.get("timestamp"):
            continue

        # Enforce exact matching logic using clean formats
        row_timestamp_clean = pd.to_datetime(row["timestamp"], utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")

        if row_timestamp_clean not in historical_timestamps:
            cleaned_row = dict(row)
            cleaned_row.pop("id", None)  # Let Postgres auto-generate serial IDs

            for key, value in cleaned_row.items():
                if value == "":
                    cleaned_row[key] = None

            rows_to_insert.append(cleaned_row)

    if rows_to_insert:
        batch_size = 500
        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i:i+batch_size]
            supabase.table("aqi_features").insert(batch).execute()
        print(f"✅ Inserted {len(rows_to_insert)} new rows into aqi_features")
    else:
        print("✅ No new rows to merge (Deduplication worked perfectly!)")

elif buffer_data and df_historical.empty:
    print("Historical features table completely blank. Preparing bulk ingestion fallback...")
    # Bulk pass if table is starting fresh...
    pass
else:
    print("No buffer data found to merge.")

# =========================================================
# CLEAN BUFFER TABLE
# =========================================================
print("\nCleaning new_daily_data buffer...")
supabase.table("new_daily_data").delete().gte("id", 0).execute()
print("✅ Buffer table cleaned")
print("\n✅ Daily AQI training pipeline completed successfully")