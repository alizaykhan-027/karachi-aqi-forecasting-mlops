import os
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

# ============================================================
# LOAD ENV
# ============================================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase credentials")

# ============================================================
# SUPABASE CONNECTION
# ============================================================
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# LOAD PRODUCTION MODELS
# ============================================================
try:
    model_24 = joblib.load("models/best_model_24h.pkl")
    model_48 = joblib.load("models/best_model_48h.pkl")
    model_72 = joblib.load("models/best_model_72h.pkl")
    print("✅ Models loaded successfully")
except Exception as e:
    raise FileNotFoundError(f"❌ Failed to load trained models from disk: {e}")

# ============================================================
# FETCH LATEST FEATURE MATRIX POINT
# ============================================================
response = (
    supabase
    .table("aqi_features")
    .select("*")
    .order("timestamp", desc=True)
    .limit(1)
    .execute()
)

if not response.data:
    raise ValueError("❌ No feature records found in aqi_features table to run inference on.")

latest = pd.DataFrame(response.data)
print(f"✅ Latest feature row fetched. Reference Time: {latest['timestamp'].iloc[0]}")

# ============================================================
# STATIC FEATURE DEFINITIONS (Must match exactly what went into .fit())
# ============================================================
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

# Safeguard feature matching structure
for col in optional_cols:
    if col in latest.columns:
        priority_features.append(col)

# Strict index selection
priority_features = [col for col in priority_features if col in latest.columns]

# Ensure any missing optional metrics or lagging points are handled safely as zeros
X = latest[priority_features].copy()
X = X.fillna(0)

# ============================================================
# PREDICTIONS
# ============================================================
pred_24 = max(0.0, float(model_24.predict(X)[0]))
pred_48 = max(0.0, float(model_48.predict(X)[0]))
pred_72 = max(0.0, float(model_72.predict(X)[0]))

print("\n" + "="*26 + " AQI FORECAST " + "="*26)
print(f"🔮 24H Outbreak Projection : {pred_24:.2f} AQI")
print(f"🔮 48H Outbreak Projection : {pred_48:.2f} AQI")
print(f"🔮 72H Outbreak Projection : {pred_72:.2f} AQI")
print("="*66)

# ============================================================
# SAVE PREDICTIONS WITH NATIVE ISO TIMESTAMPS
# ============================================================
prediction_row = {
    "timestamp": pd.to_datetime(latest["timestamp"].iloc[0]).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "aqi_24h_prediction": pred_24,
    "aqi_48h_prediction": pred_48,
    "aqi_72h_prediction": pred_72
}

supabase.table("aqi_predictions").insert(prediction_row).execute()
print("\n✅ Predictions stored successfully in database")