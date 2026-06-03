# 🌦️ SkyCast: Automated MLOps Engine for Multi-Horizon Urban Air Quality Forecasting

SkyCast is a serverless, production-grade AI system designed to forecast time-series Air Quality Index (AQI) trajectories for Karachi, Pakistan across 24-hour, 48-hour, and 72-hour future horizons. By combining automated data harvesting engines with a cloud-synchronized feature store, the framework establishes a continuous machine learning lifecycle—orchestrating near-real-time ingestion pipelines, automated daily optimization, and decoupled deployment endpoints.

---

## 🔗 Live Production Gateways

The system architecture exposes a client-facing visualization dashboard alongside remote experimentation registries:

👉 **Launch Live SkyCast Production Dashboard**
`YOUR_ACTUAL_DEPLOYMENT_URL_HERE`

👉 **Explore Remote Model Registry & Experiment Tracking Logs**
`YOUR_DAGSHUB_OR_MLFLOW_URL_HERE`

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=Streamlit\&logoColor=white)](YOUR_ACTUAL_DEPLOYMENT_URL_HERE)
[![DagsHub Tracking](https://img.shields.io/badge/DagsHub-MLflow-000000?style=for-the-badge\&logo=git\&logoColor=white)](YOUR_DAGSHUB_OR_MLFLOW_URL_HERE)

---

# 🏗️ Production System Lifecycle Architecture

The platform operates entirely via automated, modular cloud transitions to eliminate manual data or file maintenance:

```text
[ Multi-Year Environmental Ledger ]
                │
                ▼
[ Data Ingestion Sync ]
                │
                ▼
[ Supabase Cloud Feature Store ]
                │
                ▼
[ GitHub Actions MLOps Orchestrator ]
                │
                ▼
[ Serialization Engine ]
                │
                ▼
[ Interactive Streamlit Web UI ]
```

### Automated Ingestion Workflow (`hourly_pipeline.yml`)

Fetches incoming environmental trace parameters and blends them over a strict timeline using deterministic inner joins. Processed records are synchronized directly into the Supabase Cloud Feature Store.

### Automated Continuous Training (`daily_training_pipeline.yml`)

Triggers a complete retraining cycle every 24 hours. The pipeline:

* Evaluates historical model performance
* Performs hyperparameter optimization
* Registers experiments and artifacts to a remote tracking server
* Selects the best-performing model
* Automatically updates production-ready model binaries

---

# 📂 Repository Structure

```text
KARACHI-AQI-FORECASTING-PLATFORM/
│
├── .github/
│   └── workflows/
│       ├── daily_training_pipeline.yml
│       └── hourly_pipeline.yml
│
├── dashboards/
│   └── streamlit_app.py
│
├── data/
│   ├── processed/
│   │   ├── .gitkeep
│   │   └── aqi_feature_store.csv
│   │
│   └── raw/
│       ├── .gitkeep
│       └── karachi_aqi_historical.csv
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── model_registry/
│   │   └── register_model.py
│   │
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── retrain_pipeline.py
│   ├── train_model.py
│   └── upload_to_supabase.py
│
├── utils.py
├── .cache_sqlite
├── .env
├── .gitignore
├── LICENSE
└── requirements.txt
```

---

# 📊 Feature Engineering & Data Pipeline

The system transforms heterogeneous environmental data into a structured feature matrix optimized for AQI forecasting.

## Meteorological Features

* Temperature
* Relative Humidity
* Atmospheric Pressure
* Wind Speed
* Wind Direction

## Air Quality Features

* PM2.5
* PM10
* Carbon Monoxide (CO)
* Nitrogen Dioxide (NO₂)
* Ozone (O₃)
* Sulfur Dioxide (SO₂)

## Engineered Features

* Multi-hour lag variables
* Rolling means
* Rolling standard deviations
* Exponential moving averages
* AQI trend indicators
* Cyclical temporal encodings

  * `hour_sin`
  * `hour_cos`
  * `month_sin`
  * `month_cos`
* Pollution event flags
* Seasonal indicators

---

# 🏆 Model Benchmarking & Evaluation

Multiple forecasting architectures were benchmarked under identical evaluation protocols to identify the most accurate production model.

| Model                   | Horizon  | MAE  | RMSE | R² Score | Status        |
| ----------------------- | -------- | ---- | ---- | -------- | ------------- |
| Extra Trees Regressor   | 24 Hours | X.XX | X.XX | 0.XX     | 🏅 Production |
| Extra Trees Regressor   | 48 Hours | X.XX | X.XX | 0.XX     | 🏅 Production |
| Extra Trees Regressor   | 72 Hours | X.XX | X.XX | 0.XX     | 🏅 Production |
| Random Forest Regressor | 24 Hours | X.XX | X.XX | 0.XX     | Baseline      |
| XGBoost Regressor       | 24 Hours | X.XX | X.XX | 0.XX     | Baseline      |
| LightGBM Regressor      | 24 Hours | X.XX | X.XX | 0.XX     | Baseline      |
| LSTM Recurrent Network  | 24 Hours | X.XX | X.XX | 0.XX     | Baseline      |

> Replace placeholder metrics with actual model performance values.

---

# 🛡️ Explainable AI (XAI)

To improve transparency and trustworthiness, SkyCast incorporates both global and local interpretability frameworks.

## SHAP (Global Explainability)

Provides feature importance analysis across the entire dataset.

Key observations include:

* PM2.5 lag features strongly influence short-term AQI predictions.
* Carbon monoxide contributes significantly during pollution spikes.
* Temperature and pressure affect long-term atmospheric dispersion.

## LIME (Local Explainability)

Generates prediction-specific explanations directly inside the dashboard.

Features are visualized using:

* 🟧 Positive contributions (increase AQI)
* 🟦 Negative contributions (decrease AQI)

This enables users to understand why a specific forecast was generated.

---

# 🚀 Streamlit Dashboard Features

The production dashboard provides a user-friendly interface for environmental monitoring and forecasting.

### 📈 AQI Visualization

* Current AQI status
* 24-hour forecast
* 48-hour forecast
* 72-hour forecast
* AQI category overlays

### 🚨 Automated Health Advisories

Transforms numerical AQI forecasts into understandable public-health warnings.

### 🤖 Generate AI Forecast

Creates automated narrative summaries based on current environmental conditions and forecasted AQI trajectories.

### 📥 Export Results

Users can download:

* Forecast results
* Feature matrices
* Historical records
* CSV reports

---

# ⚙️ MLOps Stack

### Data Layer

* Supabase Feature Store
* Automated Data Collection
* Cloud Synchronization

### Training Layer

* Scikit-Learn
* XGBoost
* LightGBM
* TensorFlow/Keras

### Experiment Tracking

* DagsHub
* MLflow

### Deployment Layer

* Streamlit Cloud
* GitHub Actions
* Automated Retraining

---

# 🛠️ Local Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/KARACHI-AQI-FORECASTING-PLATFORM.git

cd KARACHI-AQI-FORECASTING-PLATFORM
```

## 2. Create Virtual Environment

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file in the project root directory:

```env
SUPABASE_URL="YOUR_SUPABASE_PROJECT_ENDPOINT"

SUPABASE_KEY="YOUR_SUPABASE_ACCESS_TOKEN"

DAGSHUB_TOKEN="YOUR_DAGSHUB_REPO_TOKEN"
```

---

# ▶️ Run Streamlit Dashboard

```bash
streamlit run dashboards/streamlit_app.py
```

---

# 📅 Automation Schedule

| Workflow                    | Frequency      | Purpose                                |
| --------------------------- | -------------- | -------------------------------------- |
| hourly_pipeline.yml         | Every Hour     | Data Ingestion & Feature Store Updates |
| daily_training_pipeline.yml | Every 24 Hours | Retraining & Model Selection           |
| Streamlit Deployment        | Continuous     | Real-Time Forecast Delivery            |

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Alizay Khan**

AI Engineer | Machine Learning | MLOps | Environmental Forecasting

Developed as a complete end-to-end MLOps platform for automated urban air-quality prediction and continuous model lifecycle management.

├── utils.py                            # Shared structural utility scripts and modules
├── .cache_sqlite                       # Speed optimization network cache layer
├── requirements.txt                    # Project production dependency blueprint
└── README.md                           # System operational manual

