# ============================================================
# DAGSHUB MODEL REGISTRY SCRIPT
# ============================================================

import os
import joblib
import dagshub
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# LOAD ENV VARIABLES
# ============================================================

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# ============================================================
# CHECK ENV VARIABLES
# ============================================================

if not DAGSHUB_USERNAME:
    raise ValueError("❌ DAGSHUB_USERNAME not found in environment")

if not DAGSHUB_REPO:
    raise ValueError("❌ DAGSHUB_REPO not found in environment")

if not DAGSHUB_TOKEN:
    raise ValueError("❌ DAGSHUB_TOKEN not found in environment")

# ============================================================
# AUTHENTICATION
# ============================================================

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

dagshub.auth.add_app_token(DAGSHUB_TOKEN)

# ============================================================
# INITIALIZE DAGSHUB
# ============================================================

dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)

print("✅ DagsHub initialized")

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

best_model_24h = joblib.load(
    "models/best_model_24h.pkl"
)

best_model_48h = joblib.load(
    "models/best_model_48h.pkl"
)

best_model_72h = joblib.load(
    "models/best_model_72h.pkl"
)

print("✅ Models loaded successfully")

# ============================================================
# CREATE / SET EXPERIMENT
# ============================================================

mlflow.set_experiment("AQI_Forecasting")

# ============================================================
# START MLFLOW RUN
# ============================================================

with mlflow.start_run(run_name="ExtraTrees_Final_Model"):

    # ========================================================
    # METRICS
    # ========================================================

    mlflow.log_metric("24h_MAE", 8.603)
    mlflow.log_metric("24h_RMSE", 12.063)
    mlflow.log_metric("24h_R2", 0.332)

    mlflow.log_metric("48h_MAE", 9.522)
    mlflow.log_metric("48h_RMSE", 13.119)
    mlflow.log_metric("48h_R2", 0.210)

    mlflow.log_metric("72h_MAE", 9.793)
    mlflow.log_metric("72h_RMSE", 13.184)
    mlflow.log_metric("72h_R2", 0.203)

    print("✅ Metrics logged")

    # ========================================================
    # TAGS
    # ========================================================

    mlflow.set_tags({

        "Project": "Karachi AQI Forecasting",
        "City": "Karachi",
        "Model_Type": "ExtraTreesRegressor",
        "Forecast_Horizon": "24h_48h_72h",
        "Feature_Store": "Supabase",
        "Model_Registry": "DagsHub",
        "Framework": "Scikit-Learn",
        "Data_Leakage": "No"

    })

    print("✅ Tags logged")

    # ========================================================
    # LOG MODELS
    # ========================================================

    model_info_24 = mlflow.sklearn.log_model(
        sk_model=best_model_24h,
        name="extra_trees_24h_model"
    )

    model_info_48 = mlflow.sklearn.log_model(
        sk_model=best_model_48h,
        name="extra_trees_48h_model"
    )

    model_info_72 = mlflow.sklearn.log_model(
        sk_model=best_model_72h,
        name="extra_trees_72h_model"
    )

    print("✅ Models logged successfully")

    # ========================================================
    # REGISTER MODELS
    # ========================================================

    mlflow.register_model(
        model_uri=model_info_24.model_uri,
        name="AQI_24H_ExtraTrees"
    )

    print("✅ 24H model registered")

    mlflow.register_model(
        model_uri=model_info_48.model_uri,
        name="AQI_48H_ExtraTrees"
    )

    print("✅ 48H model registered")

    mlflow.register_model(
        model_uri=model_info_72.model_uri,
        name="AQI_72H_ExtraTrees"
    )

    print("✅ 72H model registered")

# ============================================================
# FINISHED
# ============================================================

print("\n ALL MODELS SUCCESSFULLY REGISTERED TO DAGSHUB")
