# =============================================================================
# DATA COLLECTION PIPELINE
# KARACHI AQI + WEATHER DATA
# PRODUCTION READY
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

import os
import openmeteo_requests
import pandas as pd
import requests_cache

from retry_requests import retry
from datetime import datetime, timedelta

# =============================================================================
# CREATE DATA DIRECTORY
# =============================================================================
os.makedirs("data/raw", exist_ok=True)

# =============================================================================
# SETUP API CLIENT
# =============================================================================
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# =============================================================================
# LOCATION
# =============================================================================
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

# =============================================================================
# DYNAMIC DATE WINDOW
# =============================================================================
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=4)

start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

print("=" * 60)
print("FETCHING KARACHI AQI & WEATHER DATA")
print("=" * 60)
print(f"Date Window: {start_date_str} → {end_date_str}")

# =============================================================================
# PART 1: AIR QUALITY DATA
# =============================================================================
print("\n1. Fetching air quality data...")
air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

aq_metrics = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "us_aqi", "european_aqi"]

aq_params = {
    "latitude": KARACHI_LAT,
    "longitude": KARACHI_LON,
    "start_date": start_date_str,
    "end_date": end_date_str,
    "hourly": aq_metrics
}

aq_response = openmeteo.weather_api(air_quality_url, params=aq_params)[0]
aq_hourly = aq_response.Hourly()

# Safe, dynamic extraction by variable order mapping
aq_data = {}
for idx, metric in enumerate(aq_metrics):
    aq_data[metric] = aq_hourly.Variables(idx).ValuesAsNumpy()

aq_length = len(aq_hourly.Variables(0).ValuesAsNumpy())
aq_timestamps = pd.date_range(
    start=pd.to_datetime(aq_hourly.Time(), unit="s", utc=True),
    periods=aq_length,
    freq=pd.Timedelta(seconds=aq_hourly.Interval()),
    inclusive="left"
)

aq_df = pd.DataFrame({"timestamp": aq_timestamps, **aq_data})

# =============================================================================
# PART 2: WEATHER DATA (SWITCHED TO FORECAST API FOR REAL-TIME AVAILABILITY)
# =============================================================================
print("2. Fetching weather data...")
# CHANGED: Using forecast endpoint to avoid the multi-day update delay of the Archive API
weather_url = "https://api.open-meteo.com/v1/forecast"

weather_metrics = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pressure_msl", "precipitation", "cloudcover"]

weather_params = {
    "latitude": KARACHI_LAT,
    "longitude": KARACHI_LON,
    "start_date": start_date_str,
    "end_date": end_date_str,
    "hourly": weather_metrics
}

weather_response = openmeteo.weather_api(weather_url, params=weather_params)[0]
weather_hourly = weather_response.Hourly()

# Safe dynamic extraction
weather_data = {}
for idx, metric in enumerate(weather_metrics):
    weather_data[metric] = weather_hourly.Variables(idx).ValuesAsNumpy()

weather_length = len(weather_hourly.Variables(0).ValuesAsNumpy())
weather_timestamps = pd.date_range(
    start=pd.to_datetime(weather_hourly.Time(), unit="s", utc=True),
    periods=weather_length,
    freq=pd.Timedelta(seconds=weather_hourly.Interval()),
    inclusive="left"
)

# Column renaming map to match your clean schemas
rename_map = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "pressure_msl": "pressure"
}

weather_df = pd.DataFrame({"timestamp": weather_timestamps, **weather_data})
weather_df = weather_df.rename(columns=rename_map)

# =============================================================================
# PART 3: MERGE DATA
# =============================================================================
print("3. Merging AQI + Weather tables...")
merged_df = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
merged_df.insert(0, "city", "Karachi")

# =============================================================================
# PART 4: SAVE DATA
# =============================================================================
csv_filename = "data/raw/karachi_aqi_historical.csv"
merged_df.to_csv(csv_filename, index=False)

# =============================================================================
# FINAL INFORMATION
# =============================================================================
print("\n" + "=" * 60)
print("DATA COLLECTION COMPLETED")
print("=" * 60)
print(f"\n✅ File saved successfully:\n{csv_filename}")
print(f"\nDataset Information:\n{'-'*40}")
print(f"Total Rows: {len(merged_df)}")
print(f"Date Range: {merged_df['timestamp'].min()} → {merged_df['timestamp'].max()}")
print(f"{'-'*40}\n\nPreview:")
print(merged_df.tail())