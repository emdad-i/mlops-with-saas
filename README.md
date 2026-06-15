# MLOps with SaaS

An end-to-end MLOps learning project built using free SaaS tools. This repository teaches you how to train, version, deploy, monitor, and maintain machine learning models without managing cloud infrastructure.

**What is MLOps?** MLOps (Machine Learning Operations) is the practice of applying DevOps principles to ML. It bridges the gap between data scientists building models and engineers deploying them to production. This guide shows you a modern, scalable approach using industry-standard free tools.

## What You'll Learn

By following this guide, you'll:

- Train and version control ML models using GitHub
- Track experiments with Weights & Biases
- Store and manage model artifacts on Hugging Face Hub
- Build production-ready APIs with FastAPI
- Automate training and deployment with GitHub Actions
- Monitor model performance with Prometheus and Grafana Cloud
- Upload models to Hugging Face Hub for inference

## Quick Start (5 minutes)

If you want to run the project locally **right now**:

```bash
# Clone and setup
git clone <repo-url>
cd mlops-with-saas
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set environment variables (create .env file or export in shell)
export HUGGINGFACE_REPO_ID=your-username/your-model-repo
export HUGGINGFACE_HUB_TOKEN=hf_your_token_here

# Train the model
python src/train.py

# Start the API locally
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` to test the API.

---


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

You'll need free accounts on these platforms. Don't worry — they're all free tier friendly:

## GitHub (Source Control)

**Why:** Git tracks every change to your code. GitHub lets you collaborate and automate deployments.

- Sign up: https://github.com
- Create a new repository called `mlops-with-saas`
- Clone it locally: `git clone <repo-url>`

**In this project:** All code lives here. GitHub Actions will automatically run tests and deploy when you push.

---

## Weights & Biases (Experiment Tracking)

**Why:** When you train a model, you want to log metrics (accuracy, loss, etc.) and compare different training runs. W&B makes this easy.

- Sign up: https://wandb.ai
- Create a new project called `mlops-saas-lab`

**In this project:** See [src/train.py](src/train.py) — it logs training metrics to W&B with `wandb.log()`.

```python
wandb.init(project="mlops-saas-lab")
# ... train model ...
wandb.log({"accuracy": accuracy, "f1_score": f1})
```

---

## Hugging Face (Model Registry)

**Why:** After training, you need to store your model somewhere reproducible and accessible. Hugging Face Hub is a central registry for ML models.

- Sign up: https://huggingface.co
- Create two repositories:
  - `your-username/your-model-repo` (for trained models)
  - Note your username for later

**In this project:** See [src/train.py](src/train.py) — after training, it uploads the model:

```python
api = HfApi()
api.upload_file(
    path_or_fileobj="app/model.pkl",
    path_in_repo="model.pkl",
    repo_id=hf_repo_id,
    token=hf_token,
)
```

And see [app/main.py](app/main.py) — the app downloads the model from HF when it starts:

```python
model_file = hf_hub_download(
    repo_id=HUGGINGFACE_REPO_ID,
    filename="model.pkl",
)
```

---

## Render (Deployment)

**Why:** You need to host your API somewhere so it's publicly accessible. Render is free and integrates seamlessly with GitHub.

- Sign up: https://render.com
- Connect your GitHub account
- Create a Web Service (we'll do this in Step 8)

**In this project:** [app/main.py](app/main.py) is a FastAPI app that Render will run. See the `/predict` endpoint for predictions and `/metrics` for monitoring.

---

## Grafana Cloud (Monitoring)

**Why:** Once your model is live, you need to know if it's working well. Grafana dashboards visualize metrics from your API.

- Sign up: https://grafana.com/products/cloud (free tier available)
- Create a Prometheus data source pointing to your API's `/metrics` endpoint

**In this project:** [app/main.py](app/main.py) exposes Prometheus metrics at `/metrics`. Grafana will scrape this to build dashboards.

# Step 2 - Clone the Repository

```bash
git clone <your-repo-url>
cd mlops-with-saas
```

This repository contains:

- **[src/train.py](src/train.py)** — Trains a Random Forest model on the Wine dataset, logs metrics to W&B, and uploads to Hugging Face
- **[app/main.py](app/main.py)** — FastAPI app that loads the model and serves predictions
- **[requirements.txt](requirements.txt)** — All Python dependencies
- **[Dockerfile](Dockerfile)** — Container configuration for Render deployment

---

# Step 3 - Set Up Your Local Development Environment

### Create a Virtual Environment

A virtual environment keeps your project's dependencies isolated from your system Python.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `scikit-learn` — ML library for training models
- `fastapi` + `uvicorn` — Web framework for serving predictions
- `wandb` — Experiment tracking
- `huggingface_hub` — Upload/download models from Hugging Face
- `prometheus-client` — Metrics for monitoring
- `joblib` — Serialize trained models

### Set Environment Variables

Create a `.env` file in the repo root (it's already in `.gitignore`, so your secrets won't be committed):

```bash
# .env
HUGGINGFACE_REPO_ID=your-username/your-model-repo
HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxx
```

**Getting your HF token:**

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name, select "Write" access
4. Copy the token and paste it in `.env`

**Load these variables in your shell:**

```bash
source .env  # or manually: export HUGGINGFACE_REPO_ID=...
```

---

# Step 4 - Train a Model Locally

The training script is in [src/train.py](src/train.py). Here's what it does:

1. Loads the Wine dataset from scikit-learn
2. Trains a Random Forest classifier
3. Logs metrics (accuracy, precision, recall, F1) to Weights & Biases
4. Saves the model locally to `app/model.pkl`
5. **Uploads the model to Hugging Face Hub**

### Run Training

```bash
python src/train.py
```

You should see output like:

```
Model trained successfully. Test Accuracy: 0.9722
Model saved to app/model.pkl
Model uploaded to Hugging Face repo: your-username/your-model-repo/model.pkl
```

### Check Your Weights & Biases Dashboard

Go to https://wandb.ai/your-username/mlops-saas-lab and see your training metrics visualized.

### What Happened?

The training script:

```python
# From src/train.py

# 1. Initialize W&B tracking
wandb.init(project="mlops-saas-lab")

# 2. Train the model
model = RandomForestClassifier(...)
model.fit(X_train, y_train)

# 3. Log metrics
wandb.log({
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
})

# 4. Save locally
joblib.dump(model, "app/model.pkl")

# 5. Upload to Hugging Face
api = HfApi()
api.upload_file(
    path_or_fileobj="app/model.pkl",
    path_in_repo="model.pkl",
    repo_id=hf_repo_id,
    token=hf_token,
)
```

**Troubleshooting:**

- **"Skipping Hugging Face upload because HUGGINGFACE_REPO_ID is not set"** → Make sure you ran `export HUGGINGFACE_REPO_ID=...` or have it in your `.env` file and sourced it.
- **"Invalid repo_id"** → Double-check your repo name matches your Hugging Face account.
- **Permission errors** → Verify your token has "Write" access.

---

# Step 5 - Start the API Locally

Now that you have a trained model, let's serve it via an API.

The API is in [app/main.py](app/main.py). It:

- Loads the model from `app/model.pkl` (or downloads from Hugging Face if missing)
- Exposes a `/predict` endpoint for making predictions
- Exposes a `/metrics` endpoint for Prometheus to scrape

### Run the Server

```bash
uvicorn app.main:app --reload
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Test the API

Open http://localhost:8000/docs (Swagger UI) and try:

1. Click on the green `/predict` button
2. Click "Try it out"
3. Fill in some Wine features (e.g., alcohol=12.0, malic_acid=1.5, ...)
4. Click "Execute"
5. See the prediction result

Or use curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 12.0,
    "malic_acid": 1.5,
    "ash": 2.4,
    "alcalinity_of_ash": 20.0,
    "magnesium": 92.0,
    "total_phenols": 1.8,
    "flavanoids": 0.6,
    "nonflavanoid_phenols": 0.5,
    "proanthocyanidins": 0.8,
    "color_intensity": 5.5,
    "hue": 0.6,
    "od280_od315_of_diluted_wines": 1.5,
    "proline": 640.0
  }'
```

### How It Works

```python
# From app/main.py

# Load model (from local cache OR download from Hugging Face)
if os.path.exists(MODEL_LOCAL_PATH):
    model = joblib.load(MODEL_LOCAL_PATH)
elif HUGGINGFACE_REPO_ID:
    model_file = hf_hub_download(...)
    model = joblib.load(model_file)

# Define API schema
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    # ... 11 more features ...

# Serve predictions
@app.post("/predict")
def predict(data: WineFeatures):
    feature_vector = [[...all features...]]
    prediction = model.predict(feature_vector)
    return {"prediction": int(prediction[0])}

# Track metrics
prediction_requests.inc()
prediction_latency.observe(time.time() - start)
```

**Troubleshooting:**

- **"Model file not found"** → Make sure `app/model.pkl` exists (run `python src/train.py` first) OR set `HUGGINGFACE_REPO_ID` to download it.
- **"Connection refused"** → Port 8000 might be in use. Try `--port 8001` instead.

---

# Step 6 - Add Monitoring (Prometheus + Grafana)

While your API is running, Prometheus metrics are available at http://localhost:8000/metrics.

These track:

- `prediction_requests_total` — How many predictions you've made
- `prediction_latency_seconds` — How long predictions take

### In Production: Connect Grafana Cloud

Once deployed to Render (Step 7+), you can visualize these metrics:

1. Create a Grafana Cloud dashboard at https://grafana.com/products/cloud
2. Add a Prometheus datasource pointing to `https://your-app.onrender.com/metrics`
3. Build dashboards to monitor:
   - Request volume over time
   - Average prediction latency
   - Error rates (if any)

See [app/main.py](app/main.py) for the metrics implementation:

```python
from prometheus_client import Counter, Histogram

prediction_requests = Counter("prediction_requests_total", "...")
prediction_latency = Histogram("prediction_latency_seconds", "...")

# In the predict endpoint:
prediction_requests.inc()
prediction_latency.observe(time.time() - start)
```

---

# Step 7 - Commit and Push to Git

```bash
git add .
git commit -m "Initial MLOps setup with model training and FastAPI"
git push origin main
```

**Important:** Make sure `.env` is in `.gitignore` so secrets are never committed.

---

# Step 8 - Deploy to Render

Render will automatically deploy your app every time you push to GitHub.

### Connect Render to GitHub

1. Go to https://render.com and sign up
2. Click "New +" → "Web Service"
3. Select "Connect a repository" and choose your `mlops-with-saas` repo
4. Fill in the details:
   - **Name:** mlops-saas-lab (or your choice)
   - **Environment:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Add Environment Secrets

In the Render dashboard:

1. Go to your service's "Environment" tab
2. Add these environment variables:
   - `HUGGINGFACE_REPO_ID`: your-username/your-model-repo
   - `HUGGINGFACE_HUB_TOKEN`: hf_xxxxxxxxxxxx
   - `METRICS_USER`: (optional, for basic auth on `/metrics`)
   - `METRICS_PASS`: (optional, for basic auth on `/metrics`)

### Deploy

Click "Create Web Service". Render will:

1. Pull your code from GitHub
2. Install dependencies
3. Start your FastAPI app
4. Assign you a public URL like `https://mlops-saas-lab.onrender.com`

Your app is now live! 🎉

### Test Deployed API

```bash
curl https://your-app.onrender.com/docs
# or open in browser
```

---

# Step 9 - Set Up CI/CD (GitHub Actions)

Automate testing and deployment with GitHub Actions. This runs every time you push code.

Create `.github/workflows/deploy.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches:
      - main

jobs:
  test-and-deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      # Install dependencies
      - run: pip install -r requirements.txt

      # (Optional) Run tests if you have a tests/ folder
      - run: pytest tests/ || echo "No tests yet"

      # Trigger Render deployment (Render listens to GitHub)
      # No explicit step needed — Render auto-deploys on push
```

Now, every time you:

```bash
git push origin main
```

GitHub Actions will:

1. Check out your code
2. Install dependencies
3. Run tests (if any)
4. Render automatically redeploys

---

# Step 10 - Experiment: Retrain and Update

The MLOps loop is continuous:

1. **Collect new data** → Retrain `python src/train.py`
2. **Log metrics** → View in Weights & Biases
3. **Upload new model** → Hugging Face Hub
4. **Git commit** → Push to GitHub
5. **Automatic deploy** → New model live on Render
6. **Monitor** → Check Grafana dashboards

---

# Troubleshooting & Common Issues

### "HUGGINGFACE_REPO_ID is not set"

Your environment variables aren't loaded. Try:

```bash
export HUGGINGFACE_REPO_ID=your-username/your-repo
export HUGGINGFACE_HUB_TOKEN=hf_xxx
python src/train.py
```

Or create `.env` and `source` it:

```bash
source .env
python src/train.py
```

### "Model file not found"

Either:

- Run `python src/train.py` to create it locally, OR
- Set `HUGGINGFACE_REPO_ID` so the app downloads it

### "Permission denied" on model upload

Your HF token might not have Write access. Go to https://huggingface.co/settings/tokens and regenerate it with Write permissions.

### "Port already in use"

Change the port:

```bash
uvicorn app.main:app --reload --port 8001
```

### Tests are failing or metrics are wrong

Check [src/train.py](src/train.py) — modify the feature columns or target column if your dataset differs from the Wine dataset.

---

# Next Steps for Learning

- **Experiment Tracking:** Modify [src/train.py](src/train.py) to try different hyperparameters, watch W&B log the differences
- **Model Versioning:** Create multiple model repos on Hugging Face and swap between them
- **Data Pipeline:** Extend [src/train.py](src/train.py) to pull data from a database or API
- **Advanced Monitoring:** Build custom dashboards in Grafana Cloud
- **Unit Tests:** Add tests to `tests/` folder
- **Docker:** Use the included [Dockerfile](Dockerfile) to run locally as a container

---

# Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Weights & Biases Docs](https://docs.wandb.ai/)
- [Hugging Face Hub Guide](https://huggingface.co/docs/hub)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Render Deployment Guide](https://render.com/docs)
- [Prometheus + Grafana](https://grafana.com/docs/)

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
