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

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials")

# ============================================================
# SUPABASE CLIENT
# ============================================================
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Connected to Supabase")

# ============================================================
# LOAD FEATURE FILE
# ============================================================
df = pd.read_csv("data/processed/aqi_feature_store.csv")
print(f"✅ Loaded {len(df)} rows from aqi_feature_store.csv")

if df.empty:
    print("❌ No data found in file")
    exit()

# ============================================================
# FORMAT TIMESTAMP (STRICT DECI-SECOND MATCHING)
# ============================================================
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# ============================================================
# CLEAN NaN / INF (SAFE TARGET PRESERVATION)
# ============================================================
df = df.replace([np.inf, -np.inf], np.nan)

# Fill standard inputs/lags with 0, but KEEP future targets as NaN (SQL NULL)
target_cols = [col for col in df.columns if "target" in col]
feature_cols = [col for col in df.columns if col not in target_cols and col != "timestamp"]

df[feature_cols] = df[feature_cols].fillna(0)
df[target_cols] = df[target_cols].astype(object).where(df[target_cols].notna(), None)

# ============================================================
# FETCH EXISTING TIMESTAMPS FROM BUFFER TABLE (FIXED)
# ============================================================
existing_data = (
    supabase.table("new_daily_data")
    .select("timestamp")
    .execute()
)

existing_timestamps = set()

if existing_data.data:
    # Normalize whatever string format Postgres returns into standard UTC
    existing_timestamps = {
        pd.to_datetime(row["timestamp"], utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
        for row in existing_data.data
        if row.get("timestamp")
    }

print(f"Existing timestamps in new_daily_data: {len(existing_timestamps)}")

# ============================================================
# FILTER NEW ROWS ONLY (FIXED)
# ============================================================
df = df[~df["timestamp"].isin(existing_timestamps)]
print(f"New rows to upload: {len(df)}")

if df.empty:
    print("✅ No new rows found. Database is perfectly up to date!")
    exit()

# ============================================================
# CONVERT TO RECORDS
# ============================================================
records = df.to_dict(orient="records")

# ============================================================
# UPLOAD IN BATCHES
# ============================================================
chunk_size = 200
print("\nUploading to new_daily_data...")

for i in range(0, len(records), chunk_size):
    batch = records[i:i + chunk_size]
    try:
        supabase.table("new_daily_data").insert(batch).execute()
        print(f"✅ Uploaded rows {i} → {i + len(batch)}")
        time.sleep(0.5)
    except Exception as e:
        print(f"\n❌ Error at batch {i}")
        print(e)
        break

print("\n✅ Buffer upload completed")

# ============================================================
# VERIFY MAX TIMESTAMP
# ============================================================
latest = (
    supabase.table("new_daily_data")
    .select("timestamp")
    .order("timestamp", desc=True)
    .limit(1)
    .execute()
)

if latest.data:
    print(f"Latest timestamp in table: {latest.data[0]['timestamp']}")