import os
import joblib
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictRequest, PredictResponse, ModelStatusResponse

app = FastAPI(
    title="NAF NAF Sales Forecasting API",
    description="ML API for weekly sales prediction by Famille",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== LOAD MODEL ON STARTUP =====
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_model")

model       = joblib.load(os.path.join(MODELS_DIR, "ridge_model.pkl"))
scaler      = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
famille_map = joblib.load(os.path.join(MODELS_DIR, "famille_map.pkl"))
FEATURES    = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))

FAMILIES_SUPPORTED = sorted(famille_map.keys())

MODEL_METRICS = {
    "wmape": 63.62,
    "r2": 0.4645,
    "last_trained": "2026-04-07"
}

SUMMER_FAM = ['BERMUDA', 'BERMUDA/SHORT', 'SHORT', 'T-SHIRT',
              'TEE-SHIRT', 'T SHIRT SANS MANCHES', 'T SHIRT MANCHES COUR']
WINTER_FAM = ['MANTEAU', 'PULL', 'BONNETERIE', 'BONNETERIE/COIFFANT',
              'GILET', 'SWEAT', 'SWEATSHIRT']

# ===== ROUTES =====

@app.get("/")
def root():
    return {"message": "NAF NAF ML API is running 🚀", "version": "1.0.0"}


@app.get("/families")
def get_families():
    return {"families": FAMILIES_SUPPORTED}


@app.get("/model/status", response_model=ModelStatusResponse)
def model_status():
    return ModelStatusResponse(
        status="healthy",
        model_name="Ridge (alpha=1.0) — Global Model",
        wmape=MODEL_METRICS["wmape"],
        r2=MODEL_METRICS["r2"],
        last_trained=MODEL_METRICS["last_trained"],
        families_supported=FAMILIES_SUPPORTED
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.famille not in famille_map:
        raise HTTPException(
            status_code=400,
            detail=f"Famille '{req.famille}' not supported. Choose from: {FAMILIES_SUPPORTED}"
        )

    famille_encoded = famille_map[req.famille]

    # Derive calendar features automatically from year + week
    week_date      = datetime.fromisocalendar(req.year, req.week_of_year, 1)
    month          = week_date.month
    quarter        = (month - 1) // 3 + 1
    is_summer      = int(month in [6, 7, 8])
    is_winter      = int(month in [12, 1, 2])
    is_soldes      = int(month in [1, 7])
    is_high_season = int(
        (req.famille in SUMMER_FAM and is_summer) or
        (req.famille in WINTER_FAM and is_winter)
    )

    feature_row = np.array([[
        req.year, req.week_of_year, month, quarter,
        is_summer, is_winter, is_soldes, is_high_season,
        famille_encoded,
        req.lag_1, req.lag_2, req.lag_4, req.lag_52,
        req.rolling_mean_4, req.rolling_mean_12
    ]])

    scaled     = scaler.transform(feature_row)
    prediction = float(max(model.predict(scaled)[0], 0))

    return PredictResponse(
        famille=req.famille,
        year=req.year,
        week_of_year=req.week_of_year,
        predicted_quantity=round(prediction, 2)
    )
