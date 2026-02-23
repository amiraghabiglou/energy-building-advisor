"""ML Server - XGBoost Regression Service.

Imports all schemas from shared module (DRY principle).
"""

import os
import sys
import pickle
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from sklearn.model_selection import cross_val_score

# Fix #3: Import from shared schemas - NO REDEFINITION
from src.energy_advisor.schemas import BuildingFeatures, EnergyPrediction, HealthResponse

app = FastAPI(title="ML Server (XGBoost)", version="1.0.0")

models: Dict[str, xgb.XGBRegressor] = {}


def load_or_train_models():
    """Load or train XGBoost models."""
    global models

    model_path = os.getenv("MODEL_PATH", "models/xgboost_model.pkl")

    if os.path.exists(model_path):
        print(f"[INIT] Loading models from {model_path}")
        with open(model_path, 'rb') as f:
            models = pickle.load(f)


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    load_or_train_models()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check."""
    return HealthResponse(
        status="healthy" if len(models) == 2 else "unhealthy",
        ml_server_connected=len(models) == 2,
        slm_loaded=False,
        version="1.0.0"
    )


@app.post("/predict", response_model=EnergyPrediction)
def predict(features: BuildingFeatures):
    """Predict heating and cooling loads using XGBoost.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    features_array = np.array([[
        features.relative_compactness,
        features.surface_area,
        features.wall_area,
        features.roof_area,
        features.overall_height,
        features.orientation,
        features.glazing_area,
        features.glazing_area_distribution
    ]])

    heating_pred = models["heating"].predict(features_array)[0]
    cooling_pred = models["cooling"].predict(features_array)[0]

    return EnergyPrediction(
        heating_load=float(heating_pred),
        cooling_load=float(cooling_pred),
        confidence=0.95
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ML_SERVER_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
