import base64
import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from prometheus_client import make_asgi_app, Counter, Histogram
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.responses import Response

# --- Prometheus Metrics ---
prediction_requests = Counter("prediction_requests_total", "Total prediction requests")
prediction_latency = Histogram("prediction_latency_seconds", "Prediction latency")

app = FastAPI()

# Load model safely (mocked here, keep your path)
# model = joblib.load("app/model.pkl")

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanidins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.post("/predict")
def predict(data: WineFeatures):
    start = time.time()
    prediction_requests.inc()

    feature_vector = [[
        data.alcohol, data.malic_acid, data.ash, data.alcalinity_of_ash,
        data.magnesium, data.total_phenols, data.flavanoids, data.nonflavanoid_phenols,
        data.proanthocyanidins, data.color_intensity, data.hue,
        data.od280_od315_of_diluted_wines, data.proline
    ]]

    # prediction = model.predict(feature_vector)
    prediction = [1] # Temporary placeholder

    prediction_latency.observe(time.time() - start)

    return {"prediction": int(prediction[0])}


# --- Basic Auth ASGI Middleware ---
METRICS_USER = os.environ.get("METRICS_USER")
METRICS_PASS = os.environ.get("METRICS_PASS")

class BasicAuthMetrics:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # Double check that we are only intercepting HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"")
        
        try:
            if auth:
                scheme, credentials = auth.split(b" ", 1)
                if scheme.lower() == b"basic":
                    decoded = base64.b64decode(credentials).decode("utf-8")
                    username, password = decoded.split(":", 1)
                    if username == METRICS_USER and password == METRICS_PASS:
                        # Auth passed, hand over to the metrics app
                        await self.app(scope, receive, send)
                        return
        except Exception:
            pass

        # Auth failed: Return 401 Unauthorized
        response = Response(
            "Unauthorized", 
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="metrics"'}
        )
        await response(scope, receive, send)


# --- The Fix: Wrap ONLY the metrics app, then mount it ---
metrics_asgi_app = make_asgi_app()
protected_metrics_app = BasicAuthMetrics(metrics_asgi_app)

app.mount("/metrics", protected_metrics_app)