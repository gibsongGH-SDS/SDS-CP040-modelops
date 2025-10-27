from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import os

import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ---- Paths ----
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "model.pkl"
TEMPLATES_DIR = HERE / "templates"

# ---- Lifespan: load/unload resources here ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model once and attach to app.state
    try:
        app.state.model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    yield  # <-- application runs while we're yielded

    # Shutdown: clean up if needed (nothing to do here)
    # e.g., close db connections, release resources, etc.

# ---- FastAPI app ----
app = FastAPI(
    title="Car Price Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---- Schemas ----
class CarInput(BaseModel):
    manufacturer: str
    model: str
    fuel_type: str
    engine_size: float
    year_of_manufacture: int
    mileage: float

# ---- Routes ----
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # Jinja2 render keeps headers & relative paths clean
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
        headers={"Content-Type": "text/html; charset=utf-8"},
    )

@app.post("/predict")
def predict_car_price(payload: CarInput, request: Request):
    # Ensure model is present
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Derived features (mirror training pipeline assumptions)
    CURRENT_YEAR = 2025
    age = max(CURRENT_YEAR - payload.year_of_manufacture, 0)
    mileage_per_year = payload.mileage / max(age, 1)
    vintage = int(age >= 20)

    # Map to the exact training column names (with spaces) + derived
    row = {
        "Manufacturer": payload.manufacturer,
        "Model": payload.model,
        "Fuel type": payload.fuel_type,
        "Engine size": payload.engine_size,
        "Year of manufacture": payload.year_of_manufacture,
        "Mileage": payload.mileage,
        "age": age,
        "mileage_per_year": mileage_per_year,
        "vintage": vintage,
    }
    df = pd.DataFrame([row])

    try:
        pred = model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {"predicted_price_gbp": round(float(pred), 2)}

# ---- Dev entrypoint (ignored in Docker CMD) ----
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


