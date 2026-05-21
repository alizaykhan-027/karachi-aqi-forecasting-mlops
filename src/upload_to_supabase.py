import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ============================================================
# ENV VARIABLES
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ============================================================
# SUPABASE CLIENT
# ============================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

print("✅ Connected to Supabase")

# ============================================================
# LOAD FEATURE STORE CSV
# ============================================================

df = pd.read_csv(
    "data/processed/aqi_feature_store.csv"
)

print(f"✅ Loaded {len(df)} feature rows")

# ============================================================
# FORMAT TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df["timestamp"] = (
    df["timestamp"]
    .dt.strftime("%Y-%m-%dT%H:%M:%S")
)

# ============================================================
# CLEAN NaN / INF
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.fillna(0)

# ============================================================
# FETCH EXISTING BUFFER TIMESTAMPS
# ============================================================

existing_data = (
    supabase.table("new_daily_data")
    .select("timestamp")
    .execute()
)

existing_timestamps = set()

if existing_data.data:
    existing_timestamps = {
        str(row["timestamp"])
        for row in existing_data.data
        if row.get("timestamp")
    }

print(f"Existing buffer timestamps: {len(existing_timestamps)}")

# ============================================================
# FILTER NEW ROWS ONLY
# ============================================================

df = df[
    ~df["timestamp"].astype(str).isin(existing_timestamps)
]

print(f"New rows to upload: {len(df)}")

if df.empty:
    print("No new rows found")
    exit()

# ============================================================
# CONVERT TO RECORDS
# ============================================================

records = df.to_dict(
    orient="records"
)

# ============================================================
# UPLOAD IN BATCHES TO BUFFER TABLE
# ============================================================

chunk_size = 200

print("\nUploading feature store to new_daily_data...")

for i in range(0, len(records), chunk_size):

    batch = records[i:i + chunk_size]

    try:

        supabase.table(
            "new_daily_data"
        ).insert(batch).execute()

        print(
            f"✅ Uploaded rows "
            f"{i} → {i + len(batch)}"
        )

        time.sleep(0.5)

    except Exception as e:

        print(f"\n❌ Error at batch {i}")
        print(e)
        break

print("\n Buffer upload completed")
