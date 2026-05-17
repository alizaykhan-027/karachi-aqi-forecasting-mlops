import os
import joblib
import pandas as pd
from supabase import create_client

# ============================================================
# ENV VARIABLES
# ============================================================

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
# REMOVE NON-FEATURE COLUMNS
# ============================================================

drop_cols = [

    "timestamp",
    "city",

    "target_24h",
    "target_48h",
    "target_72h"
]

X = latest.drop(
    columns=[c for c in drop_cols if c in latest.columns]
)

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
