from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time

from prometheus_client import make_asgi_app
from prometheus_client import Counter
from prometheus_client import Histogram

prediction_requests = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

app = FastAPI()

model = joblib.load("app/model.pkl")

# 1. Define the 13 features expected by the wine dataset
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
        data.alcohol,
        data.malic_acid,
        data.ash,
        data.alcalinity_of_ash,
        data.magnesium,
        data.total_phenols,
        data.flavanoids,
        data.nonflavanoid_phenols,
        data.proanthocyanidins,
        data.color_intensity,
        data.hue,
        data.od280_od315_of_diluted_wines,
        data.proline
    ]]

    prediction = model.predict(feature_vector)

    prediction_latency.observe(
        time.time() - start
    )

    return {
        "prediction": int(prediction[0])
    }

app.mount(
    "/metrics",
    make_asgi_app()
)