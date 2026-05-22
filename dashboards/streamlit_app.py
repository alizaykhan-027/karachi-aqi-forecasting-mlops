import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client
from dotenv import load_dotenv, find_dotenv 
import dagshub  # 👈 Added DagsHub integration

# =========================================================
# 1. PAGE SETUP & CONFIGURATION
# =========================================================
load_dotenv(find_dotenv(usecwd=True))
st.set_page_config(
    page_title="Karachi AQI MLOps Dashboard",
    page_icon="🌬️",
    layout="wide"
)

# Fetch Environment Configurations
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

DAGSHUB_REPO_OWNER = os.environ.get("DAGSHUB_USERNAME")
DAGSHUB_REPO_NAME = os.environ.get("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN")

@st.cache_resource
def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.sidebar.error("⚠️ Missing Supabase Environment Credentials.")
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# Initialize DagsHub Authentication Configuration
if DAGSHUB_TOKEN:
    os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN

# =========================================================
# 2. REMOTE MODEL REGISTRY FETCHERS (DAGSHUB)
# =========================================================
@st.cache_resource(ttl=3600)  # Cache loaded models for 1 hour to optimize bandwidth
def load_remote_model(horizon):
    """
    Dynamically pulls the specified horizon champion model file from DagsHub Storage
    and loads its serialized array directly into memory.
    """
    if not DAGSHUB_REPO_OWNER or not DAGSHUB_REPO_NAME:
        return None
        
    remote_path = f"models/best_model_{horizon}.pkl"
    local_download_dir = "models"
    local_path = os.path.join(local_download_dir, f"best_model_{horizon}.pkl")
    
    # Ensure local path infrastructure is available
    os.makedirs(local_download_dir, exist_ok=True)
    
    try:
        # If the file hasn't been cached locally yet, download it from DagsHub
        if not os.path.exists(local_path):
            repo_url = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
            
            # Streams the binary from DagsHub using your security token
            dagshub.download_url(
                repo_url=repo_url,
                remote_path=remote_path,
                local_path=local_path
            )
            
        if os.path.exists(local_path):
            return joblib.load(local_path)
    except Exception as e:
        st.sidebar.warning(f"Failed loading {horizon} model from DagsHub: {e}")
    return None

# =========================================================
# 3. CACHED DATABASE FETCHERS (PAGINATED)
# =========================================================
@st.cache_data(ttl=1800)  
def fetch_feature_store():
    if not supabase:
        return pd.DataFrame()
    try:
        all_data = []
        offset = 0
        batch_size = 1000
        while True:
            res = supabase.table("aqi_features").select("*").range(offset, offset + batch_size - 1).execute()
            batch = res.data
            if not batch:
                break
            all_data.extend(batch)
            offset += batch_size
            if len(batch) < batch_size:
                break
        
        df = pd.DataFrame(all_data)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error reading feature store: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)  
def fetch_production_model_metrics():
    if not supabase:
        return pd.DataFrame()
    try:
        res = supabase.table("model_performance").select("*").eq("is_production", True).execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        st.error(f"Error loading model metrics: {e}")
        return pd.DataFrame()

# Global Data Load
df_features = fetch_feature_store()
df_metrics = fetch_production_model_metrics()

# Application Tabs Layout
tab_forecast, tab_eda, tab_metrics, tab_explain = st.tabs([
    "🔮 3-Day Forecast", 
    "📈 Feature Store Trends (EDA)", 
    "🏆 Production Tournament Standings",
    "🧠 Feature Importance (SHAP)"
])

# =========================================================
# TAB 1: LIVE FORECASTS & PUBLIC ADVISORIES
# =========================================================
with tab_forecast:
    st.title("🛰️ Real-Time Karachi Air Quality Forecast")
    
    if df_features.empty:
        st.warning("Feature database is currently empty or unreachable.")
    else:
        latest_feature_row = df_features.iloc[[-1]]
        latest_timestamp = latest_feature_row["timestamp"].values[0]
        st.caption(f"Last pipeline sync state: **{pd.Timestamp(latest_timestamp).strftime('%B %d, %Y - %I:%M %p')}**")
        
        priority_features = [
            "pm2_5", "us_aqi", "pm25_log", "hour_sin", "hour_cos", "day_sin", "day_cos",
            "month_sin", "month_cos", "is_weekend", "season", "aqi_lag_1h", "aqi_lag_6h",
            "aqi_lag_24h", "aqi_rolling_avg_6h", "aqi_rolling_avg_24h", "aqi_change_rate_24h",
            "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
            "pm25_lag_48", "pm25_roll_mean_3", "pm25_roll_mean_6", "pm25_roll_mean_12", "pm25_roll_mean_24",
            "pm25_roll_std_24", "pm25_roll_min_24", "pm25_roll_max_24", "pm25_median_24", "pm25_ema_3",
            "pm25_ema_6", "pm25_ema_12", "pm25_ema_24", "pm25_volatility_24", "pm25_cumulative_24",
            "high_pollution_flag"
        ]
        optional_cols = ["pm10", "pm_ratio", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "temperature", "humidity", "wind_speed", "pressure"]
        
        active_features = [col for col in priority_features if col in df_features.columns]
        for col in optional_cols:
            if col in df_features.columns:
                active_features.append(col)
                
        X_live = latest_feature_row[active_features]
        
        horizons = ["24h", "48h", "72h"]
        predictions = {}
        
        for h in horizons:
            # Dynamically fetch the model directly from your DagsHub Registry
            model = load_remote_model(h)
            if model is not None:
                try:
                    predictions[h] = round(float(model.predict(X_live)[0]), 1)
                except Exception:
                    predictions[h] = np.nan
            else:
                predictions[h] = np.nan

        # Metric Cards Layout
        col1, col2, col3 = st.columns(3)
        with col1:
            val_24 = f"{predictions['24h']}" if not np.isnan(predictions['24h']) else "DagsHub Model missing"
            st.metric(label="📆 24-Hour Forecast (AQI)", value=val_24)
        with col2:
            val_48 = f"{predictions['48h']}" if not np.isnan(predictions['48h']) else "DagsHub Model missing"
            st.metric(label="📅 48-Hour Forecast (AQI)", value=val_48)
        with col3:
            val_72 = f"{predictions['72h']}" if not np.isnan(predictions['72h']) else "DagsHub Model missing"
            st.metric(label="📆 72-Hour Forecast (AQI)", value=val_72)
            
        st.markdown("---")
        
        # Automated Health Protection Advisories
        st.subheader("🚨 Health Warnings & Public Advisories")
        valid_preds = [v for v in predictions.values() if not np.isnan(v)]
        
        if valid_preds:
            max_aqi = max(valid_preds)
            if max_aqi > 300:
                st.error(f"🔴 **CRITICAL HAZARDOUS ADVISORY:** Predicted AQI peaks at **{max_aqi}**. Avoid all outdoor activities; N95 air masks are strongly advised.")
            elif max_aqi > 200:
                st.warning(f"⚠️ **VERY UNHEALTHY RISK WARNING:** Forecast reaches an unhealthy ceiling of **{max_aqi}**. Children, elderly, and sensitive groups should stay indoors.")
            elif max_aqi > 150:
                st.info(f"🔵 **UNHEALTHY AIR QUALITY LEVEL:** Elevated AQI projected (**{max_aqi}**). Consider reducing heavy outdoor exercise.")
            else:
                st.success(f"🟢 **HEALTHY ENVIRONMENT NOTE:** Maximum projected 3-day AQI is **{max_aqi}**, which falls within safe public exposure parameters.")
        else:
            st.info("💡 Synchronizing registry assets... Verify that your `.pkl` files exist inside your DagsHub project repositories.")

# =========================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
with tab_eda:
    st.title("📈 Feature Store Insights & Historical Analysis")
    st.markdown("Analytical review tracking physical measurements and engineering data directly from your master table.")
    
    if not df_features.empty:
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            st.subheader("Recent Pollutant Distribution Trends")
            fig, ax = plt.subplots(figsize=(10, 4.5))
            tail_df = df_features.tail(168)  
            ax.plot(tail_df["timestamp"], tail_df["pm2_5"], color="#ff4b4b", label="$PM_{2.5}$ Index")
            if "nitrogen_dioxide" in tail_df.columns:
                ax.plot(tail_df["timestamp"], tail_df["nitrogen_dioxide"], color="#0068c9", label="$NO_2$ Concentration")
            ax.set_ylabel("Measurement Scale")
            ax.set_xlabel("Timeline Index")
            ax.legend()
            st.pyplot(fig)
            
        with col_eda2:
            st.subheader("Pollution Flags vs Volatility Distribution")
            fig2, ax2 = plt.subplots(figsize=(10, 4.5))
            if "pm25_volatility_24" in df_features.columns and "high_pollution_flag" in df_features.columns:
                sns.boxplot(data=df_features, x="high_pollution_flag", y="pm25_volatility_24", ax=ax2, palette="muted")
                ax2.set_xticklabels(["Standard Conditions", "High Pollution Triggered"])
            else:
                sns.histplot(df_features["us_aqi"], kde=True, ax=ax2, color="purple")
                ax2.set_title("Overall Historical US AQI Frequency Profile")
            st.pyplot(fig2)
    else:
        st.info("Awaiting connection to feature tables to map distributions.")

# =========================================================
# TAB 3: PRODUCTION TOURNAMENT STANDINGS
# =========================================================
with tab_metrics:
    st.title("🏆 Live Production Champion Metrics")
    st.markdown("Displays active metadata evaluation profiles compiled directly from your serverless master tournament loop.")
    
    if not df_metrics.empty:
        st.dataframe(df_metrics[["horizon", "model_name", "mae", "rmse", "r2"]], use_container_width=True)
        
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        sns.barplot(data=df_metrics, x="horizon", y="rmse", hue="model_name", ax=ax3, palette="Set2")
        ax3.set_title("Root Mean Squared Error (RMSE) Baseline Breakdown per Horizon Segment")
        st.pyplot(fig3)
    else:
        st.info("No active metadata entries discovered in your `model_performance` database yet.")

# =========================================================
# TAB 4: GLOBAL INTERPRETABILITY (SHAP Feature Importance)
# =========================================================
with tab_explain:
    st.title("🧠 Global Model Attribution & Weights")
    st.markdown("Exposes feature importance metrics directly from your trained production files.")
    
    model_24 = load_remote_model("24h")
    if model_24 is not None:
        try:
            if hasattr(model_24, "feature_importances_"):
                importances = model_24.feature_importances_
                
                feat_imp_df = pd.DataFrame({
                    "Feature Name": active_features,
                    "Relative Importance Score": importances
                }).sort_values("Relative Importance Score", ascending=False).head(10)
                
                fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                sns.barplot(data=feat_imp_df, x="Relative Importance Score", y="Feature Name", ax=ax_shap, color="#00f5d4")
                ax_shap.set_title("Top 10 Feature Contributor Attributions (Champion Model)")
                st.pyplot(fig_shap)
                st.caption("💡 Variables on top dictate major prediction movements when computing target values.")
            else:
                st.info("The active champion model does not have built-in feature importance tracking.")
        except Exception as e:
            st.error(f"Could not calculate importances dynamically: {e}")
    else:
        st.caption("Fetching DagsHub artifact context. Displaying generic fallback reference hierarchy:")
        mock_features = ["pm2_5", "aqi_lag_24h", "pm25_lag_1", "temperature", "humidity"]
        mock_weights = [0.45, 0.25, 0.15, 0.10, 0.05]
        
        fig_shap, ax_shap = plt.subplots(figsize=(10, 4))
        ax_shap.barh(mock_features, mock_weights, color="#4ea8de")
        ax_shap.invert_yaxis()
        st.pyplot(fig_shap)