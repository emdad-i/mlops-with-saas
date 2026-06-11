from fastapi import FastAPI
import joblib

from prometheus_client import make_asgi_app
from prometheus_client import Counter

prediction_requests = Counter(
    "prediction_requests_total",
    "Total prediction requests"
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
    
    # Optional: prediction_requests.inc() if you are using Prometheus
    
    # 2. Convert the incoming Pydantic data into the 2D array the model expects
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
    
    # 3. Make the prediction
    prediction = model.predict(feature_vector)
    
    return {
        "prediction": int(prediction[0])
    }

app.mount(
    "/metrics",
    make_asgi_app()
)