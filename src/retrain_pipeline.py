import os
import joblib
import pandas as pd

from supabase import create_client
from sklearn.ensemble import ExtraTreesRegressor
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# ENV
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
# FETCH DATA
# ============================================================

all_data = []

page_size = 1000
start = 0

while True:

    response = (
        supabase
        .table("aqi_features")
        .select("*")
        .range(start, start + page_size - 1)
        .execute()
    )

    batch = response.data

    if not batch:
        break

    all_data.extend(batch)

    start += page_size

df = pd.DataFrame(all_data)

print(f"✅ Loaded {len(df)} rows")

# ============================================================
# CLEANING
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

drop_cols = [
    "timestamp",
    "city"
]

targets = [
    "target_24h",
    "target_48h",
    "target_72h"
]

feature_cols = [

    c for c in df.columns

    if c not in drop_cols + targets
]

X = df[feature_cols]

# ============================================================
# TRAIN MODELS
# ============================================================

for target in targets:

    y = df[target]

    model = ExtraTreesRegressor(

        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    filename = f"models/{target}.pkl"

    joblib.dump(model, filename)

    print(f"✅ Saved {filename}")
