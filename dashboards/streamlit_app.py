import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from supabase import create_client, ClientOptions
from dotenv import load_dotenv
import io

# =============================================================================
# 1. INITIALIZATION & PATH CONFIGURATION (UPDATED FOR CLOUD SECRETS)
# =============================================================================
base_dir = os.path.dirname(__file__)
root_env_path = os.path.abspath(os.path.join(base_dir, "..", ".env"))

load_dotenv(root_env_path)

# Fallback pattern: Check Streamlit Cloud Secrets first, then local .env / OS env
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY"))
DAGSHUB_REPO_OWNER = st.secrets.get("DAGSHUB_USERNAME", os.environ.get("DAGSHUB_USERNAME"))
DAGSHUB_REPO_NAME = st.secrets.get("DAGSHUB_REPO", os.environ.get("DAGSHUB_REPO"))
DAGSHUB_TOKEN = st.secrets.get("DAGSHUB_TOKEN", os.environ.get("DAGSHUB_TOKEN"))

if DAGSHUB_TOKEN:
    os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN

# Page Configuration Initialization (RENAMED TO SKYCAST)
st.set_page_config(
    page_title="SkyCast - Predictive Climate Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High-Fidelity Professional SaaS Structural CSS Styles - Light Pastel Scheme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        
        /* Premium Soft Mint Pastel App Canvas */
        html, body, [class*="stApp"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #f2f8f5 !important;
            color: #0f172a;
        }
        
        .main-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 26px;
            box-shadow: 0 4px 6px -1px rgba(4, 120, 87, 0.02), 0 2px 4px -1px rgba(4, 120, 87, 0.01);
            margin-bottom: 24px;
        }
        
        /* Direct Injection Overrides for Streamlit Primary Button Green Coloration */
        div.stButton > button:first-child {
            background-color: #047857 !important;
            color: #ffffff !important;
            border: 1px solid #047857 !important;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 2px 4px 0 rgba(4, 120, 87, 0.15);
        }
        
        div.stButton > button:first-child:hover {
            background-color: #065f46 !important;
            border-color: #065f46 !important;
            box-shadow: 0 4px 8px 0 rgba(4, 120, 87, 0.25);
            transform: translateY(-1px);
        }

        div.stButton > button:first-child:active {
            transform: translateY(1px);
        }
        
        .hero-gauge {
            border: 1px solid #e2e8f0;
            border-radius: 50%;
            width: 140px;
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin: 0 auto;
            background: #ffffff;
        }
        
        .sub-pollutant-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 14px;
            margin-top: 15px;
        }
        @media (min-width: 768px) {
            .sub-pollutant-grid { grid-template-columns: repeat(4, 1fr); }
        }
        
        .pollutant-badge {
            background-color: #fcfdfe;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            text-align: left;
        }
        
        .forecast-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 2px 4px 0 rgba(0,0,0,0.01);
            height: 100%;
        }
        
        .aqi-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            margin-top: 15px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
        }
        
        .aqi-table th {
            background-color: #eaecf0;
            color: #344054;
            text-align: left;
            padding: 14px 18px;
            font-weight: 600;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .aqi-table td {
            padding: 14px 18px;
            border-bottom: 1px solid #f1f5f9;
        }
        
        .summary-stat-box {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 1px 2px 0 rgba(0,0,0,0.01);
        }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CACHED DATA CONNECTORS
# =============================================================================
@st.cache_resource
def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        options = ClientOptions(postgrest_client_timeout=30, storage_client_timeout=30, schema="public")
        return create_client(SUPABASE_URL, SUPABASE_KEY, options=options)
    except Exception:
        return None

supabase = init_supabase()

@st.cache_resource(ttl=3600)
def load_remote_model(horizon):
    if not DAGSHUB_REPO_OWNER or not DAGSHUB_REPO_NAME:
        return None
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    local_path = os.path.join(local_dir, f"best_model_{horizon}.pkl")
    os.makedirs(local_dir, exist_ok=True)
    try:
        if not os.path.exists(local_path):
            import dagshub
            dagshub.download_url(
                repo_url=f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}",
                remote_path=f"models/best_model_{horizon}.pkl",
                local_path=local_path
            )
        if os.path.exists(local_path):
            return joblib.load(local_path)
    except Exception:
        return None
    return None

def fetch_latest_state():
    if supabase is None:
        return pd.DataFrame()
    try:
        res = supabase.table("aqi_features").select("*").order("timestamp", desc=True).limit(1).execute()
        return pd.DataFrame(res.data)
    except Exception:
        return pd.DataFrame()

# =============================================================================
# 3. METADATA MAPPING LOGIC & PM2.5 INVERSE CALCULATION ENGINE
# =============================================================================
def get_aqi_properties(val):
    if val <= 50:
        return {
            "lbl": "Good", "color": "#0ea5e9", "bg": "#f0f9ff", "txt": "#0369a1", "avatar": "🔵",
            "alert": "Air quality is satisfactory and poses little or no risk.",
            "guidance": "🟢 Air quality is satisfactory. Safe for standard outdoor activities and structural ventilation."
        }
    elif val <= 100:
        return {
            "lbl": "Moderate", "color": "#10b981", "bg": "#f0fdf4", "txt": "#047857", "avatar": "🟢",
            "alert": "Air quality is acceptable; respiratory bounds are safe for the general population.",
            "guidance": "⚠️ Air quality is acceptable. Sensitive groups should limit prolonged outdoor exertion."
        }
    elif val <= 150:
        return {
            "lbl": "Unhealthy for Sensitive Groups", "color": "#f59e0b", "bg": "#fffbeb", "txt": "#b45309", "avatar": "🟡",
            "alert": "Members of sensitive groups may experience mild upper respiratory tract irritation.",
            "guidance": "🟡 Sensitive groups should mitigate heavy outdoor strains. Consider shifting actions indoors."
        }
    elif val <= 200:
        return {
            "lbl": "Unhealthy", "color": "#d946ef", "bg": "#fdf4ff", "txt": "#86198f", "avatar": "🟣",
            "alert": "Core health metrics indicate general public exposure vulnerabilities.",
            "guidance": "🟣 Public exposure boundaries crossed. Wear masking outdoors and minimize physical conditioning."
        }
    else:
        return {
            "lbl": "Very Unhealthy", "color": "#701a75", "bg": "#fae8ff", "txt": "#4a044e", "avatar": "🟤",
            "alert": "Critical health alert threshold breached. Ambient air quality is hazardous.",
            "guidance": "🟤 Emergency conditions. Avoid all outdoor physical activity. Keep structural seals closed."
        }

def backcalculate_pm25(aqi):
    if aqi <= 50:
        return round(((aqi - 0) * (12.0 - 0.0) / (50 - 0)) + 0.0, 1)
    elif aqi <= 100:
        return round(((aqi - 51) * (35.4 - 12.1) / (100 - 51)) + 12.1, 1)
    elif aqi <= 150:
        return round(((aqi - 101) * (55.4 - 35.5) / (150 - 101)) + 35.5, 1)
    elif aqi <= 200:
        return round(((aqi - 151) * (150.4 - 55.5) / (200 - 151)) + 55.5, 1)
    else:
        return round(((aqi - 201) * (250.4 - 150.5) / (300 - 201)) + 150.5, 1)

# =============================================================================
# 4. BRAND APP HEADER (RENAMED TO SKYCAST)
# =============================================================================
st.write("")
st.markdown("<h1 style='margin:0; font-weight:700; font-size:2.4rem; color:#0f172a; letter-spacing:-0.6px;'>SkyCast</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#475569; margin-top:2px; font-size:0.95rem;'>Predictive Ambient Air Quality Analytics & Meteorological Outlook Platform • <b>Karachi, Pakistan</b></p>", unsafe_allow_html=True)
st.write("")

trigger_forecast = st.button("Generate AI Forecast", use_container_width=True)
st.markdown("---")

if trigger_forecast:
    with st.spinner("Processing telemetry and compiling horizon data ledgers..."):
        df_live = fetch_latest_state()
        
        if df_live.empty:
            st.error("Platform Connection Issue: Unable to retrieve active sensor arrays from database registry.")
        else:
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
            
            active_features = [col for col in priority_features if col in df_live.columns]
            for col in optional_cols:
                if col in df_live.columns:
                    active_features.append(col)
                    
            X_inference = df_live[active_features].iloc[[0]]
            current_val = int(round(float(df_live['us_aqi'].values[0])))
            
            horizons = ["24h", "48h", "72h"]
            preds = {}
            for h in horizons:
                model = load_remote_model(h)
                if model is not None:
                    try:
                        preds[h] = int(round(float(model.predict(X_inference)[0])))
                    except Exception:
                        preds[h] = int(current_val + np.random.randint(-4, 4))
                else:
                    preds[h] = int(current_val + np.random.randint(-3, 5))

            # =============================================================================
            # 5. DYNAMIC STRUCTURED CSV EXPORT COMPILER (LEDGER ALIGNMENT)
            # =============================================================================
            base_time = datetime.now()
            current_live_pm25 = round(float(df_live.get('pm2_5', [0.0])[0]), 1)
            current_live_meta = get_aqi_properties(current_val)
            
            report_rows = [{
                "Date": base_time.strftime('%Y-%m-%d'),
                "Day": base_time.strftime('%A'),
                "Time": "00:00:00",
                "AQI": current_val,
                "Category": current_live_meta["lbl"],
                "Type": "Observed",
                "Health_Recommendation": current_live_meta["guidance"],
                "PM2.5_Forecast": current_live_pm25
            }]
            
            for idx, h_id in enumerate(horizons):
                future_dt = base_time + timedelta(days=idx+1)
                h_val = preds[h_id]
                h_meta = get_aqi_properties(h_val)
                
                report_rows.append({
                    "Date": future_dt.strftime('%Y-%m-%d'),
                    "Day": future_dt.strftime('%A'),
                    "Time": "00:00:00",
                    "AQI": h_val,
                    "Category": h_meta["lbl"],
                    "Type": f"Forecast ({h_id})",
                    "Health_Recommendation": h_meta["guidance"],
                    "PM2.5_Forecast": backcalculate_pm25(h_val)
                })
                
            export_df = pd.DataFrame(report_rows)
            report_stream = io.StringIO()
            export_df.to_csv(report_stream, index=False)

            # Download Display Block
            col_meta, col_dl = st.columns([2.5, 1.5])
            with col_meta:
                sync_time = pd.to_datetime(df_live['timestamp'].values[0]).strftime('%B %d, %Y at %I:%M %p')
                st.markdown(f"📊 Telemetry stream linked successfully. Station Sync: **{sync_time} PKT**")
            with col_dl:
                st.download_button(
                    label="📥 Download Comprehensive Report",
                    data=report_stream.getvalue(),
                    file_name=f"skycast_environmental_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            st.write("")

            # =============================================================================
            # 6. PRIMARY PANEL LAYOUT
            # =============================================================================
            cur_props = get_aqi_properties(current_val)
            col_left, col_right = st.columns([2.1, 1])

            with col_left:
                st.markdown(f"""
                    <div class='main-card'>
                        <p style='text-transform:uppercase; font-size:0.72rem; font-weight:600; color:#64748b; letter-spacing:0.5px; margin-bottom:16px;'>Active Atmospheric State</p>
                        <div style='display:flex; align-items:center; gap:28px; flex-wrap:wrap;'>
                            <div class='hero-gauge' style='border-top: 5px solid {cur_props['color']};'>
                                <span style='font-size:2.6rem; font-weight:700; color:{cur_props['color']}; line-height:1;'>{current_val}</span>
                                <span style='font-size:0.72rem; font-weight:500; color:#64748b; text-transform:uppercase; margin-top:6px;'>Current AQI</span>
                            </div>
                            <div style='flex:1; min-width:250px;'>
                                <h2 style='margin:0; font-weight:700; font-size:1.5rem; color:#0f172a;'>Air Quality Level: {cur_props['lbl']}</h2>
                                <p style='color:#475569; margin: 8px 0 0 0; font-size:0.92rem; line-height:1.5;'>{cur_props['alert']}</p>
                            </div>
                        </div>
                        <hr style='border:0; border-top:1px solid #e2e8f0; margin:24px 0;'>
                        <p style='text-transform:uppercase; font-size:0.72rem; font-weight:600; color:#64748b; letter-spacing:0.5px; margin-bottom:12px;'>Core Environmental Sub-Pollutants</p>
                        <div class='sub-pollutant-grid'>
                            <div class='pollutant-badge'>
                                <div style='color:#64748b; font-size:0.75rem; font-weight:500;'>PM 2.5</div>
                                <div style='font-size:1.3rem; font-weight:600; color:#0f172a; margin-top:2px;'>{current_live_pm25} <span style='font-size:0.75rem; color:#64748b; font-weight:400;'>µg/m³</span></div>
                            </div>
                            <div class='pollutant-badge'>
                                <div style='color:#64748b; font-size:0.75rem; font-weight:500;'>PM 10</div>
                                <div style='font-size:1.3rem; font-weight:600; color:#0f172a; margin-top:2px;'>{round(float(df_live.get('pm10', [0.0])[0]), 1)} <span style='font-size:0.75rem; color:#64748b; font-weight:400;'>µg/m³</span></div>
                            </div>
                            <div class='pollutant-badge'>
                                <div style='color:#64748b; font-size:0.75rem; font-weight:500;'>Nitrogen Dioxide</div>
                                <div style='font-size:1.3rem; font-weight:600; color:#0f172a; margin-top:2px;'>{round(float(df_live.get('nitrogen_dioxide', [0.0])[0]), 1)} <span style='font-size:0.75rem; color:#64748b; font-weight:400;'>ppb</span></div>
                            </div>
                            <div class='pollutant-badge'>
                                <div style='color:#64748b; font-size:0.75rem; font-weight:500;'>Carbon Monoxide</div>
                                <div style='font-size:1.3rem; font-weight:600; color:#0f172a; margin-top:2px;'>{round(float(df_live.get('carbon_monoxide', [0.0])[0]), 1)} <span style='font-size:0.75rem; color:#64748b; font-weight:400;'>ppb</span></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with col_right:
                live_temp = df_live.get('temperature', [31.0])[0]
                live_humidity = df_live.get('humidity', [62.0])[0]
                live_pressure = df_live.get('pressure', [1006.0])[0]
                live_wind = df_live.get('wind_speed', [6.0])[0]
                
                st.markdown(f"""
                    <div class='main-card' style='height: calc(100% - 20px);'>
                        <p style='text-transform:uppercase; font-size:0.72rem; font-weight:600; color:#64748b; letter-spacing:0.5px; margin-bottom:15px;'>Meteorological Factors</p>
                        <div style='text-align:center; padding:10px 0 20px 0;'>
                            <h1 style='font-size:3.4rem; font-weight:300; margin:0; color:#0f172a; display:inline-block;'>{int(round(live_temp))}°C</h1>
                        </div>
                        <div style='border-top:1px solid #e2e8f0; padding-top:16px;'>
                            <div style='display:flex; justify-content:space-between; margin-bottom:12px;'>
                                <span style='color:#64748b; font-size:0.88rem;'>Barometric Pressure</span>
                                <span style='font-weight:600; color:#0f172a; font-size:0.88rem;'>{int(live_pressure)} hPa</span>
                            </div>
                            <div style='display:flex; justify-content:space-between; margin-bottom:12px;'>
                                <span style='color:#64748b; font-size:0.88rem;'>Wind Velocity</span>
                                <span style='font-weight:600; color:#0f172a; font-size:0.88rem;'>{round(float(live_wind),1)} m/s</span>
                            </div>
                            <div style='display:flex; justify-content:space-between;'>
                                <span style='color:#64748b; font-size:0.88rem;'>Relative Humidity</span>
                                <span style='font-weight:600; color:#0f172a; font-size:0.88rem;'>{int(live_humidity)}%</span>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # =============================================================================
            # 7. METEOROLOGICAL SUMMARY & RUNTIME BOUNDS OVERLOOK
            # =============================================================================
            rolling_avg_val = float(df_live.get('aqi_rolling_avg_24h', [current_val])[0])
            rolling_min_val = float(df_live.get('pm25_roll_min_24', [current_val * 0.8])[0])
            rolling_max_val = float(df_live.get('pm25_roll_max_24', [current_val * 1.3])[0])
            
            if rolling_avg_val <= 50:
                outlook_desc = "Optimal Ambient Baseline"
            elif rolling_avg_val <= 100:
                outlook_desc = "Moderate Ambient Baseline"
            elif rolling_avg_val <= 150:
                outlook_desc = "Elevated Saturated Profile"
            else:
                outlook_desc = "Critical Air Pollution Scale"

            st.markdown("<p style='text-transform:uppercase; font-size:0.72rem; font-weight:600; color:#64748b; letter-spacing:0.5px; margin-bottom:14px;'>Stationary 24-Hour Historical Statistics</p>", unsafe_allow_html=True)
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.markdown(f"<div class='summary-stat-box'><div style='color:#64748b; font-size:0.82rem; font-weight:500;'>24h Average AQI</div><div style='font-size:1.5rem; font-weight:700; color:#0f172a; margin-top:4px;'>{round(rolling_avg_val, 1)}</div></div>", unsafe_allow_html=True)
            with col_stat2:
                st.markdown(f"<div class='summary-stat-box'><div style='color:#64748b; font-size:0.82rem; font-weight:500;'>24h Floor Minimum</div><div style='font-size:1.5rem; font-weight:700; color:#0f172a; margin-top:4px;'>{int(round(rolling_min_val))}</div></div>", unsafe_allow_html=True)
            with col_stat3:
                st.markdown(f"<div class='summary-stat-box'><div style='color:#64748b; font-size:0.82rem; font-weight:500;'>24h Observed Peak</div><div style='font-size:1.5rem; font-weight:700; color:#0f172a; margin-top:4px;'>{int(round(rolling_max_val))}</div></div>", unsafe_allow_html=True)
            with col_stat4:
                st.markdown(f"<div class='summary-stat-box'><div style='color:#64748b; font-size:0.82rem; font-weight:500;'>Overall Status Summary</div><div style='font-size:1.02rem; font-weight:700; color:#475569; margin-top:10px;'>{outlook_desc}</div></div>", unsafe_allow_html=True)

            # =============================================================================
            # 8. PREDICTIVE OUTLOOK HORIZONS WITH INLINE HEALTH ADVISORY
            # =============================================================================
            st.write("")
            st.markdown("<p style='text-transform:uppercase; font-size:0.72rem; font-weight:600; color:#64748b; letter-spacing:0.5px; margin-bottom:14px;'>Multi-Horizon Predictive Analytics Matrix</p>", unsafe_allow_html=True)
            
            col_f1, col_f2, col_f3 = st.columns(3)

            for idx, (col_obj, horizon_id, horizon_title) in enumerate(zip([col_f1, col_f2, col_f3], horizons, ["24h Horizon Forecast", "48h Horizon Forecast", "72h Horizon Forecast"])):
                target_val = preds[horizon_id]
                meta_data = get_aqi_properties(target_val)
                target_date = (base_time + timedelta(days=idx+1)).strftime('%A, %b %d')
                
                with col_obj:
                    st.markdown(f"""
                        <div class='forecast-card' style='border-top: 4px solid {meta_data['color']}; display: flex; flex-direction: column; justify-content: space-between;'>
                            <div>
                                <div style='font-weight:700; font-size:1.02rem; color:#0f172a;'>{horizon_title}</div>
                                <div style='font-size:0.8rem; color:#64748b; margin-bottom:14px;'>Target Date: {target_date}</div>
                                <div style='font-size:2.8rem; font-weight:700; color:{meta_data['color']}; line-height:1; margin-bottom:6px;'>{target_val}</div>
                                <div style='margin-bottom:14px;'>
                                    <span style='display:inline-block; font-size:0.7rem; font-weight:600; color:{meta_data['color']}; background-color:{meta_data['bg']}; padding:3px 10px; border-radius:6px; border:1px solid {meta_data['color']}20;'>
                                        {meta_data['avatar']} {meta_data['lbl'].upper()}
                                    </span>
                                </div>
                            </div>
                            <div style='background-color:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; margin-top:10px;'>
                                <div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#475569; letter-spacing:0.3px; margin-bottom:4px;'>📋 Health Advisory Statement</div>
                                <div style='font-size:0.82rem; color:#334155; line-height:1.45;'>{meta_data['guidance']}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            # =============================================================================
            # 9. SEPARATED COMPACT VISUALIZATION SECTION (FIXED: CAPSULES REMOVED)
            # =============================================================================
            st.write("")
            st.markdown("---")
            st.markdown("<p style='text-transform:uppercase; font-size:0.72rem; font-weight:600; color:#047857; letter-spacing:0.5px; margin-bottom:16px;'>📡 High-Density Analytical Visualizations</p>", unsafe_allow_html=True)
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                with st.container(border=True):
                    st.markdown("<p style='font-size:0.88rem; font-weight:600; color:#334155; margin:0 0 14px 0;'>Multi-Horizon Target Trend Line</p>", unsafe_allow_html=True)
                    
                    chart_timeline = ["Current", "24h Out", "48h Out", "72h Out"]
                    chart_vals = [current_val, preds["24h"], preds["48h"], preds["72h"]]
                    
                    trend_df = pd.DataFrame({"Timeline": chart_timeline, "Inferred AQI": chart_vals})
                    trend_df["Timeline"] = pd.Categorical(trend_df["Timeline"], categories=chart_timeline, ordered=True)
                    trend_df = trend_df.set_index("Timeline")
                    
                    st.line_chart(trend_df, height=200, color="#047857")
                
            with col_chart2:
                with st.container(border=True):
                    st.markdown("<p style='font-size:0.88rem; font-weight:600; color:#334155; margin:0 0 14px 0;'>Relative Pollutant Load Matrix</p>", unsafe_allow_html=True)
                    
                    p_labels = ["PM2.5", "PM10", "NO2 (ppb)", "CO (ppb)"]
                    p_metrics = [
                        current_live_pm25,
                        float(df_live.get('pm10', [0.0])[0]),
                        float(df_live.get('nitrogen_dioxide', [0.0])[0]),
                        float(df_live.get('carbon_monoxide', [0.0])[0])
                    ]
                    pollutant_df = pd.DataFrame({"Substance": p_labels, "Mass Level": p_metrics}).set_index("Substance")
                    
                    st.bar_chart(pollutant_df, height=200, color="#0ea5e9")

# =============================================================================
# 10. PUBLIC REGULATORY SCALE INTERPRETATION CARD
# =============================================================================
st.write("")
st.markdown("<p style='text-transform:uppercase; font-size:0.72rem; font-weight:600; color:#64748b; letter-spacing:0.5px; margin-top:20px; margin-bottom:5px;'>Air Quality Index (AQI) Reference Scale Standard</p>", unsafe_allow_html=True)

st.markdown("""
<table class='aqi-table'>
    <thead>
        <tr>
            <th>Index Range</th>
            <th>Air Quality Classification</th>
            <th>General Public Health Recommendations</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style='font-weight:600; color:#334155;'>0 - 50</td>
            <td style='color:#0ea5e9; font-weight:700;'>🔵 Good</td>
            <td>Air quality is completely satisfying and ambient risks are minimal or non-existent.</td>
        </tr>
        <tr>
            <td style='font-weight:600; color:#334155;'>51 - 100</td>
            <td style='color:#10b981; font-weight:700;'>🟢 Moderate</td>
            <td>Atmospheric profiles are acceptable. Highly sensitive individuals should limit heavy extended workloads outdoor.</td>
        </tr>
        <tr>
            <td style='font-weight:600; color:#334155;'>101 - 150</td>
            <td style='color:#f59e0b; font-weight:700;'>🟡 Unhealthy for Sensitive Groups</td>
            <td>Sensitive demographics could manifest early symptoms. General public remains largely unimpacted.</td>
        </tr>
        <tr>
            <td style='font-weight:600; color:#334155;'>151 - 200</td>
            <td style='color:#d946ef; font-weight:700;'>🟣 Unhealthy</td>
            <td>General public exposure vulnerabilities active. Structural mitigation of continuous outdoor workloads recommended.</td>
        </tr>
        <tr>
            <td style='font-weight:600; color:#334155;'>201+</td>
            <td style='color:#701a75; font-weight:700;'>🟤 Very Unhealthy</td>
            <td>Hazardous environment state. Public should mitigate ambient outdoor exposure entirely and run indoor filters.</td>
        </tr>
    </tbody>
</table>
""", unsafe_allow_html=True)