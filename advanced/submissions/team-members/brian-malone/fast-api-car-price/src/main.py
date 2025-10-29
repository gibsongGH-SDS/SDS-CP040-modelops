from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

api = FastAPI(
    title="Car Price Prediction API",
    version="1.0.0"
)

model = None

@api.on_event("startup")
async def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "/app/models/model.pkl")
    logger.info(f"Starting model loading from {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise

class CarFeatures(BaseModel):
    Manufacturer: str
    Model: str
    Fuel_type: str = Field(alias="Fuel type")
    Engine_size: float = Field(alias="Engine size")
    Year_of_manufacture: int = Field(alias="Year of manufacture")
    Mileage: float

@api.get("/health")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "healthy", "model_loaded": model is not None}

@api.get("/metadata")
async def metadata():
    logger.debug("Metadata requested")
    return {
        "model_name": "XGBoost Car Price Predictor",
        "version": "1.0.0",
        "last_updated": "2024-10-12",
        "features": [
            "Manufacturer",
            "Model",
            "Fuel type",
            "Engine size",
            "Year of manufacture",
            "Mileage"
        ],
        "derived_features": [
            "age",
            "mileage_per_year",
            "vintage"
        ],
        "target": "price (GBP)"
    }

@api.post("/predict")
async def predict(features: CarFeatures):
    logger.info(f"Prediction request received for {features.Manufacturer} {features.Model}")

    if model is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        CURRENT_YEAR = 2025
        age = max(CURRENT_YEAR - features.Year_of_manufacture, 0)
        mileage_per_year = features.Mileage / max(age, 1)
        vintage = int(age >= 20)
        
        row = {
            "Manufacturer": features.Manufacturer,
            "Model": features.Model,
            "Fuel type": features.Fuel_type,
            "Engine size": features.Engine_size,
            "Year of manufacture": features.Year_of_manufacture,
            "Mileage": features.Mileage,
            "age": age,
            "mileage_per_year": mileage_per_year,
            "vintage": vintage,
        }
        df = pd.DataFrame([row])
        prediction = model.predict(df)[0]

        logger.info(f"Prediction successful: £{prediction:.2f} for {features.Manufacturer} {features.Model}")
        return {"predicted_price_gbp": float(prediction)}

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

