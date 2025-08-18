from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Load artifacts
model = joblib.load("crop_recommendation_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

app = FastAPI(title="Crop Recommendation API", version="1.0")

# CORS (so the HTML file can call the API from your browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/")
def root():
    return {"message": "Crop Recommendation API is running"}

@app.post("/predict")
def predict_crop(data: CropRequest):
    X = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]], dtype=float)
    pred_id = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = label_encoder.classes_.tolist()
    top3_idx = np.argsort(proba)[::-1][:3]
    return {
        "predicted_crop": classes[pred_id],
        "top3": [{"crop": classes[i], "probability": float(proba[i])} for i in top3_idx]
    }





