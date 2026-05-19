import os
import joblib
import pandas as pd

from dotenv import load_dotenv
from supabase import create_client

# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ============================================================
# SUPABASE
# ============================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# ============================================================
# LOAD MODELS
# ============================================================

model_24 = joblib.load(
    "models/best_model_24h.pkl"
)

model_48 = joblib.load(
    "models/best_model_48h.pkl"
)

model_72 = joblib.load(
    "models/best_model_72h.pkl"
)

print("✅ Models loaded")

# ============================================================
# FETCH LATEST FEATURES
# ============================================================

response = (
    supabase
    .table("aqi_features")
    .select("*")
    .order("timestamp", desc=True)
    .limit(1)
    .execute()
)

latest = pd.DataFrame(response.data)

print("✅ Latest feature row fetched")

# ============================================================
# EXACT TRAINING FEATURES
# ============================================================

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

    # FLAG
    "high_pollution_flag"
]

# ============================================================
# OPTIONAL FEATURES
# ============================================================

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

    if col in latest.columns:
        priority_features.append(col)

# ============================================================
# KEEP ONLY TRAINED FEATURES
# ============================================================

priority_features = [

    col for col in priority_features
    if col in latest.columns
]

X = latest[priority_features]

# ============================================================
# PREDICTIONS
# ============================================================

pred_24 = model_24.predict(X)[0]

pred_48 = model_48.predict(X)[0]

pred_72 = model_72.predict(X)[0]

print("\n========== AQI FORECAST ==========")

print(f"24H Prediction: {pred_24:.2f}")

print(f"48H Prediction: {pred_48:.2f}")

print(f"72H Prediction: {pred_72:.2f}")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_row = {

    "timestamp": latest["timestamp"].iloc[0],

    "aqi_24h_prediction": float(pred_24),

    "aqi_48h_prediction": float(pred_48),

    "aqi_72h_prediction": float(pred_72)
}

supabase.table(
    "aqi_predictions"
).insert(prediction_row).execute()

print("\n✅ Predictions stored in Supabase")
