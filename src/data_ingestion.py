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

# =============================================================================
# CREATE DATA DIRECTORY
# =============================================================================

os.makedirs("data/raw", exist_ok=True)

# =============================================================================
# SETUP API CLIENT
# =============================================================================

cache_session = requests_cache.CachedSession(
    ".cache",
    expire_after=3600
)

retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2
)

openmeteo = openmeteo_requests.Client(
    session=retry_session
)

# =============================================================================
# LOCATION
# =============================================================================

KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

print("=" * 60)
print("FETCHING KARACHI AQI & WEATHER DATA")
print("=" * 60)

# =============================================================================
# PART 1: AIR QUALITY DATA
# =============================================================================

print("\n1. Fetching air quality data...")

air_quality_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

aq_params = {

    "latitude": KARACHI_LAT,
    "longitude": KARACHI_LON,

    "start_date": "2023-01-01",
    "end_date": "2026-04-30",

    "hourly": [

        "pm10",
        "pm2_5",

        "carbon_monoxide",
        "nitrogen_dioxide",

        "ozone",
        "sulphur_dioxide",

        "us_aqi",
        "european_aqi"
    ]
}

aq_response = openmeteo.weather_api(
    air_quality_url,
    params=aq_params
)[0]

aq_hourly = aq_response.Hourly()

aq_length = len(
    aq_hourly
    .Variables(0)
    .ValuesAsNumpy()
)

aq_timestamps = pd.date_range(

    start=pd.to_datetime(
        aq_hourly.Time(),
        unit="s",
        utc=True
    ),

    periods=aq_length,

    freq=pd.Timedelta(
        seconds=aq_hourly.Interval()
    ),

    inclusive="left"
)

aq_df = pd.DataFrame({

    "timestamp": aq_timestamps,

    "pm10":
        aq_hourly
        .Variables(0)
        .ValuesAsNumpy(),

    "pm2_5":
        aq_hourly
        .Variables(1)
        .ValuesAsNumpy(),

    "carbon_monoxide":
        aq_hourly
        .Variables(2)
        .ValuesAsNumpy(),

    "nitrogen_dioxide":
        aq_hourly
        .Variables(3)
        .ValuesAsNumpy(),

    "ozone":
        aq_hourly
        .Variables(4)
        .ValuesAsNumpy(),

    "sulphur_dioxide":
        aq_hourly
        .Variables(5)
        .ValuesAsNumpy(),

    "us_aqi":
        aq_hourly
        .Variables(6)
        .ValuesAsNumpy(),

    "european_aqi":
        aq_hourly
        .Variables(7)
        .ValuesAsNumpy()
})

# =============================================================================
# PART 2: WEATHER DATA
# =============================================================================

print("2. Fetching weather data...")

weather_url = (
    "https://archive-api.open-meteo.com/v1/archive"
)

weather_params = {

    "latitude": KARACHI_LAT,
    "longitude": KARACHI_LON,

    "start_date": "2023-01-01",
    "end_date": "2026-04-30",

    "hourly": [

        "temperature_2m",
        "relative_humidity_2m",

        "wind_speed_10m",
        "pressure_msl",

        "precipitation",
        "cloudcover"
    ]
}

weather_response = openmeteo.weather_api(
    weather_url,
    params=weather_params
)[0]

weather_hourly = weather_response.Hourly()

weather_length = len(
    weather_hourly
    .Variables(0)
    .ValuesAsNumpy()
)

weather_timestamps = pd.date_range(

    start=pd.to_datetime(
        weather_hourly.Time(),
        unit="s",
        utc=True
    ),

    periods=weather_length,

    freq=pd.Timedelta(
        seconds=weather_hourly.Interval()
    ),

    inclusive="left"
)

weather_df = pd.DataFrame({

    "timestamp": weather_timestamps,

    "temperature":
        weather_hourly
        .Variables(0)
        .ValuesAsNumpy(),

    "humidity":
        weather_hourly
        .Variables(1)
        .ValuesAsNumpy(),

    "wind_speed":
        weather_hourly
        .Variables(2)
        .ValuesAsNumpy(),

    "pressure":
        weather_hourly
        .Variables(3)
        .ValuesAsNumpy(),

    "precipitation":
        weather_hourly
        .Variables(4)
        .ValuesAsNumpy(),

    "cloudcover":
        weather_hourly
        .Variables(5)
        .ValuesAsNumpy()
})

# =============================================================================
# PART 3: MERGE DATA
# =============================================================================

print("3. Merging AQI + Weather tables...")

merged_df = pd.merge(

    aq_df,
    weather_df,

    on="timestamp",
    how="inner"
)

merged_df.insert(
    0,
    "city",
    "Karachi"
)

# =============================================================================
# PART 4: SAVE DATA
# =============================================================================

csv_filename = (
    "data/raw/karachi_aqi_historical.csv"
)

merged_df.to_csv(
    csv_filename,
    index=False
)

# =============================================================================
# FINAL INFORMATION
# =============================================================================

print("\n" + "=" * 60)

print("DATA COLLECTION COMPLETED")

print("=" * 60)

print(f"\n✅ File saved successfully:")
print(csv_filename)

print("\nDataset Information:")
print("-" * 40)

print(f"Total Rows: {len(merged_df)}")

print(
    f"Date Range: "
    f"{merged_df['timestamp'].min()} "
    f"→ "
    f"{merged_df['timestamp'].max()}"
)

print(
    f"Average Temperature: "
    f"{merged_df['temperature'].mean():.2f} °C"
)

print(
    f"Average US AQI: "
    f"{merged_df['us_aqi'].mean():.2f}"
)

print("-" * 40)

print("\nPreview:")
print(merged_df.head())
