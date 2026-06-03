# karachi-aqi-forecasting-mlops
# 🌦️ SkyCast: Automated MLOps Engine for Multi-Horizon Urban Air Quality Forecasting

SkyCast is a serverless, production-grade AI system designed to forecast time-series Air Quality Index (AQI) trajectories for complex urban environments across 24-hour, 48-hour, and 72-hour future horizons. By combining automated data harvesting engines with a cloud-synchronized feature store, the framework establishes a continuous machine learning lifecycle—orchestrating near-real-time ingestion pipelines, automated daily optimization, and decoupled deployment endpoints.

---

## 🔗 Live Production Gateways
The system architecture exposes a client-facing visualization dashboard alongside remote experimentation registries:

👉 **[Launch Live SkyCast Production Dashboard](YOUR_ACTUAL_DEPLOYMENT_URL_HERE)** 

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](YOUR_ACTUAL_DEPLOYMENT_URL_HERE)

---

## 🏗️ Production System Lifecycle Architecture
The platform operates entirely via automated, modular cloud transitions to eliminate manual data or file maintenance:

```text
[ Multi-Year Environmental Ledger ] ──> [ Data Ingestion Sync ] ──> [ Supabase Cloud Feature Store ]
                                                                                  │
[ Interactive Streamlit Web UI ] <── [ Serialization Engine ] <── [ GitHub Actions MLOps Orchestrator ] ```text

Automated Ingestion Workflow (hourly_pipeline.yml): Fetches incoming trace parameters and blends them over a strict timeline via a deterministic inner merge, publishing streaming frames straight to a Supabase Cloud Feature Store.

Automated Continuous Training (daily_training_pipeline.yml): Triggers an automated validation run every 24 hours. The engine evaluates historical metrics, runs a hyperparameter grid tuning loop, registers artifacts to a remote tracking server, and dynamically overwrites local binaries with the winning production weights. 
