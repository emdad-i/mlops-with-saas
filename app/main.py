from fastapi import FastAPI
import joblib

from prometheus_client import Counter

prediction_requests = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)



app = FastAPI()

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: dict):

    prediction_requests.inc()

    prediction = model.predict(
        [[data["feature1"], data["feature2"]]]
    )

    return {
        "prediction": int(prediction[0])
    }

from prometheus_client import make_asgi_app

app.mount(
    "/metrics",
    make_asgi_app()
)