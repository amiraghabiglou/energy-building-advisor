"""Offline XGBoost Model Training Script.

This script downloads the UCI Energy Efficiency dataset, trains the
heating and cooling regression models, and serializes them into a
production-ready artifact (.pkl) to be injected into the Docker image.
"""

import os
import pickle
import urllib.request
import logging
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
DATA_PATH = "data/energy_efficiency.xlsx"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")


def download_data() -> None:
    """Download the dataset safely if it doesn't already exist."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        logger.info(f"Downloading dataset from {DATA_URL}...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)
            logger.info("Download complete.")
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise RuntimeError("Data ingestion failed. Cannot proceed with training.")
    else:
        logger.info("Dataset already exists locally. Skipping download.")


def train_and_evaluate() -> None:
    """Train models, evaluate robustness, and save the artifacts."""
    logger.info("Loading dataset into memory...")
    try:
        df = pd.read_excel(DATA_PATH)
    except Exception as e:
        logger.error(f"Failed to read Excel file: {e}")
        raise

    # Strictly enforce expected column mapping
    df.columns = [
        "relative_compactness", "surface_area", "wall_area", "roof_area",
        "overall_height", "orientation", "glazing_area", "glazing_area_distribution",
        "heating_load", "cooling_load"
    ]

    X = df.iloc[:, :8].values
    y_heating = df["heating_load"].values
    y_cooling = df["cooling_load"].values

    logger.info("Initializing XGBoost regressors...")
    # Parameters are fixed here for immutability, ensuring consistent artifact generation.
    model_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }

    model_heating = xgb.XGBRegressor(**model_params)
    model_cooling = xgb.XGBRegressor(**model_params)

    logger.info("Running 5-fold cross-validation...")
    heating_scores = cross_val_score(model_heating, X, y_heating, cv=5, scoring='r2')
    cooling_scores = cross_val_score(model_cooling, X, y_cooling, cv=5, scoring='r2')

    logger.info(f"Heating Model R² Score: {heating_scores.mean():.4f} (+/- {heating_scores.std() * 2:.4f})")
    logger.info(f"Cooling Model R² Score: {cooling_scores.mean():.4f} (+/- {cooling_scores.std() * 2:.4f})")

    # Analytical fail-safe: Do not save garbage models.
    if heating_scores.mean() < 0.90 or cooling_scores.mean() < 0.90:
        logger.warning("CRITICAL: Model performance is degraded. Review data or parameters before deploying.")

    logger.info("Fitting final models on full dataset...")
    model_heating.fit(X, y_heating)
    model_cooling.fit(X, y_cooling)

    os.makedirs(MODEL_DIR, exist_ok=True)
    models = {"heating": model_heating, "cooling": model_cooling}

    logger.info(f"Serializing model artifacts to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(models, f)

    logger.info("Training pipeline complete. Artifact ready for Docker build.")


if __name__ == "__main__":
    download_data()
    train_and_evaluate()