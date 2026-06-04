# 🌦️ SkyCast: Automated MLOps Engine for Multi-Horizon Karachi AQI Forecasting

SkyCast is a serverless, production-grade AI system designed to forecast Air Quality Index (AQI) levels for Karachi, Pakistan across 24-hour, 48-hour, and 72-hour future horizons. By combining automated data collection, cloud-based storage, machine learning retraining, and real-time visualization, the platform establishes a complete end-to-end MLOps lifecycle for urban air quality forecasting.

---

## 🔗 Live Production Gateway

👉 **🚀 [Live Dashboard](https://karachi-aqi-forecasting-mlops-yh4qpwm7xacy4r6yf4be5e.streamlit.app/)**

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://karachi-aqi-forecasting-mlops-yh4qpwm7xacy4r6yf4be5e.streamlit.app/)

---

# 🏗️ Production System Architecture

The platform operates through automated cloud-based workflows that continuously collect data, update feature stores, retrain models, and serve predictions to end users.

```text
[ Historical AQI & Weather Data ]
                 │
                 ▼
[ Automated Data Collection Pipeline ]
                 │
                 ▼
[ Supabase Cloud Feature Store ]
                 │
                 ▼
[ Feature Processing & Dataset Builder ]
                 │
                 ▼
[ Model Training Pipeline ]
                 │
                 ▼
[ DagsHub Model Registry (Experiment Tracking & Versioning) ]
                 │
                 ▼
[ Model Serialization (.pkl artifacts) ]
                 │
                 ▼
[ Streamlit Dashboard Deployment ]
```


### Automated Data Ingestion

The ingestion workflow continuously gathers environmental and weather-related information, processes incoming records, and synchronizes them to the Supabase Feature Store.

### Automated Model Retraining

A scheduled GitHub Actions workflow retrains the forecasting model every 24 hours using the latest available data. The best-performing model is automatically saved and used for future predictions.

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
│   │   └── aqi_feature_store.csv
│   └── raw/
│       └── karachi_aqi_historical.csv
│
├── models/
├── notebooks/
├── reports/
│
├── src/
│   ├── data_ingestion.py
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
# 🚀 Dashboard Features

The production dashboard provides an intuitive interface for monitoring and forecasting urban air quality.

### 📈 AQI Forecasting

* Current AQI monitoring
* 24-hour AQI prediction
* 48-hour AQI prediction
* 72-hour AQI prediction
* Historical AQI trend visualization

### 🚨 Health Advisory Alerts

Automatically converts AQI values into easy-to-understand health warnings based on AQI severity levels.

### 🤖 AI Forecast Summary

Generates concise natural-language summaries explaining current environmental conditions and future AQI trends.

### 📥 Export Results

Users can download forecast results and processed datasets in CSV format for further analysis.

---

# ⚙️ MLOps Technology Stack

### Data Layer

* Supabase Feature Store
* Automated Data Collection Pipelines

### Machine Learning Layer

* Python
* Scikit-Learn


### Automation Layer

* GitHub Actions
* Scheduled Retraining Workflows

### Deployment Layer

* Streamlit Cloud

---

# 🛠️ Local Installation

## 1. git clone
https://github.com/alizaykhan-027/karachi-aqi-forecasting-mlops.git
cd karachi-aqi-forecasting-mlops

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
```

---

# ▶️ Run Streamlit Dashboard

```bash
streamlit run dashboards/streamlit_app.py
```

---

# 📅 Automation Schedule

| Workflow                    | Frequency      | Purpose                                 |
| --------------------------- | -------------- | --------------------------------------- |
| hourly_pipeline.yml         | Every Hour     | Data Collection & Feature Store Updates |
| daily_training_pipeline.yml | Every 24 Hours | Model Retraining & Deployment           |

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Alizay Khan**

Developed as part of an end-to-end MLOps project for automated urban air quality forecasting and continuous machine learning lifecycle management.

**Program:** 10Pearls Shine Internship 2026 (Cohort 8)

**Track:** AI & MLOps

**Organization:** 10Pearls

**Project:** SkyCast – Multi-Horizon AQI Forecasting Platform


