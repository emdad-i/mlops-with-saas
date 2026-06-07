# Mlops with SaaS

An end-to-end MLOps learning project built using free SaaS tools. The goal is to learn how to train, version, deploy, monitor, and maintain machine learning models without managing cloud infrastructure.

## Objectives

This project demonstrates:

- Data versioning
- Experiment tracking
- Model versioning
- CI/CD automation
- API deployment
- Monitoring and observability
- Basic MLOps best practices

---

# Architecture

```text
┌─────────────┐
│   GitHub    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ GitHub      │
│ Actions     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Train Model │
└──────┬──────┘
       │
       ├──────────────► Weights & Biases
       │                 (Experiment Tracking)
       │
       ▼
┌─────────────┐
│ Model       │
│ Artifact    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ HuggingFace │
│ Hub         │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ FastAPI     │
│ Application │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Render      │
│ Deployment  │
└──────┬──────┘
       │
       ├──────────► API Endpoint
       │
       └──────────► Metrics Endpoint
                          │
                          ▼
                   Grafana Cloud
```

---

# SaaS Services Used

| Service | Purpose | Free Tier |
|----------|----------|----------|
| GitHub | Source control | Yes |
| GitHub Actions | CI/CD | Yes |
| Weights & Biases | Experiment tracking | Yes |
| Hugging Face Hub | Model registry | Yes |
| Render | API hosting | Yes |
| Grafana Cloud | Monitoring | Yes |

---

# Project Structure

```text
mlops-with-saas

├── app/
│   ├── main.py
│   ├── model.pkl
│   └── schemas.py
│
├── src/
│   ├── train.py
│   ├── preprocess.py
│   └── evaluate.py
│
├── data/
│
├── tests/
│
├── requirements.txt
├── render.yaml
├── Dockerfile
├── README.md
│
└── .github/
    └── workflows/
        └── deploy.yml
```

---

# Step 1 - Create Accounts

Create free accounts for:

## GitHub

https://github.com

Used for:

- Source control
- CI/CD
- Project management

---

## Weights & Biases

https://wandb.ai

Used for:

- Experiment tracking
- Metrics comparison
- Model evaluation history

---

## Hugging Face

https://huggingface.co

Used for:

- Model storage
- Model versioning
- Model sharing

---

## Render

https://render.com

Used for:

- FastAPI deployment
- Public API endpoints

---

## Grafana Cloud

https://grafana.com/products/cloud

Used for:

- Monitoring
- Metrics dashboards
- Alerting

---

# Step 2 - Create Repository

Create repository:

```text
mlops-with-saas
```

Clone locally:

```bash
git clone <repo-url>
cd mlops-with-saas
```

---

# Step 3 - Install Dependencies

Create virtual environment:

```bash
python -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install \
fastapi \
uvicorn \
scikit-learn \
pandas \
joblib \
wandb \
prometheus-client \
huggingface_hub
```

Freeze:

```bash
pip freeze > requirements.txt
```

---

# Step 4 - Train a Model

Example:

```python
from sklearn.ensemble import RandomForestClassifier
import wandb

wandb.init(project="mlops-saas-lab")

model = RandomForestClassifier()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

wandb.log({
    "accuracy": accuracy
})
```

Save model:

```python
import joblib

joblib.dump(
    model,
    "app/model.pkl"
)
```

---

# Step 5 - Upload Model to Hugging Face

Create token:

```text
Settings
→ Access Tokens
```

Login:

```bash
huggingface-cli login
```

Upload:

```python
from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="app/model.pkl",
    path_in_repo="model.pkl",
    repo_id="username/model-registry"
)
```

---

# Step 6 - Build FastAPI Endpoint

Example:

```python
from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: dict):

    prediction = model.predict(
        [[data["feature1"], data["feature2"]]]
    )

    return {
        "prediction": int(prediction[0])
    }
```

---

# Step 7 - Add Monitoring Metrics

Install:

```bash
pip install prometheus-client
```

Metrics:

```python
from prometheus_client import Counter

prediction_requests = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)
```

Increment:

```python
prediction_requests.inc()
```

Expose endpoint:

```python
from prometheus_client import make_asgi_app

app.mount(
    "/metrics",
    make_asgi_app()
)
```

Endpoints:

```text
POST /predict
GET /metrics
```

---

# Step 8 - Deploy to Render

Push code:

```bash
git add .
git commit -m "initial deployment"
git push
```

In Render:

```text
New Web Service
→ Connect GitHub
→ Select Repository
```

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Deploy.

Example URL:

```text
https://mlops-saas-lab.onrender.com
```

---

# Step 9 - Verify API

Swagger UI:

```text
https://your-app.onrender.com/docs
```

Prediction:

```text
POST /predict
```

Metrics:

```text
GET /metrics
```

---

# Step 10 - Connect Grafana Cloud

Create Grafana Cloud account.

Create dashboard.

Add Prometheus datasource.

Configure scraping of:

```text
https://your-app.onrender.com/metrics
```

Monitor:

- Request volume
- Prediction count
- Error count
- Response latency

---

# Step 11 - GitHub Actions CI/CD

Create:

```text
.github/workflows/deploy.yml
```

Example:

```yaml
name: CI

on:
  push:
    branches:
      - main

jobs:
  tests:

    runs-on: ubuntu-latest

    steps:

      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5

      - run: pip install -r requirements.txt

      - run: pytest
```

Every push automatically:

- Runs tests
- Validates code
- Triggers Render deployment

---

# Future Improvements

## Beginner

- Add more models
- Add model versioning
- Add feature validation

## Intermediate

- Add DVC
- Add MLflow
- Add scheduled retraining

## Advanced

- Add Prefect
- Add Feast feature store
- Add OpenTelemetry traces
- Add drift detection
- Add automated rollback

---

# Skills Demonstrated

- Machine Learning
- API Development
- Model Deployment
- CI/CD
- Monitoring
- Experiment Tracking
- Model Registry
- Cloud-native MLOps
- Production ML Workflows
