# Create a FastAPI application to predict sales price of a car (in GBP) based on vehicle characteristics.
# Intended for use in a production ModelOps API.

# Import libraries
import joblib                               # For loading the trained model
import os                                   # For file path management
import pandas as pd
from fastapi import FastAPI                 # Main FastAPI class
from fastapi import Request                 # For handling requests
from fastapi import HTTPException           # For error handling
from pydantic import BaseModel, Field       # For data validation
from fastapi.staticfiles import StaticFiles # To serve static files
from fastapi.responses import FileResponse  # To serve HTML files

app = FastAPI()  # Do NOT set docs_url=None or openapi_url=None

# Mount static directory to serve images and HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at the root URL
@app.get("/")
def read_root():
    return FileResponse("static/index.html")


# Add a pydantic model for the car features
class CarFeatures(BaseModel):
    Manufacturer: str
    Model: str
    Fuel_type: str = Field(alias="Fuel type")
    Engine_size: float = Field(alias="Engine size")
    Year_of_manufacture: int = Field(alias="Year of manufacture")
    Mileage: float

# Load the trained model ONCE when the application starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(MODEL_PATH)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Add /metadata endpoint
@app.get("/metadata")
def metadata():
    return {
        "model_name": "car-price-predictor", 
        "version": "1.0.0",
        "mlops_engineer": "Jackie CW Vescio",
        "description": "A model to predict car prices based on various features."
    }


# Add /predict endpoint
@app.post("/predict")
def predict(car_features: CarFeatures):
    try:
        CURRENT_YEAR = 2025
        car_age = max(CURRENT_YEAR - car_features.Year_of_manufacture, 0)  # Ensure non-negative age
        mileage_per_year = car_features.Mileage / max(car_age, 1)  # Avoid division by zero
        vintage = int(car_age >= 20)

        # Prepare row dictionary - collect all car features and engineered features into a single structure. 
        row = {
            "Manufacturer": car_features.Manufacturer.strip(),
            "Model": car_features.Model.strip(),
            "Engine size": car_features.Engine_size,
            "Fuel type": car_features.Fuel_type.strip(),
            "Year of manufacture": car_features.Year_of_manufacture,
            "Mileage": car_features.Mileage,
            "age": car_age,
            "mileage_per_year": mileage_per_year,
            "vintage": vintage
        }

        # Builds a table with your input data, just like the model expects
        df = pd.DataFrame([row])

        print("\n--- DATAFRAME SENT TO MODEL ---")
        print(df)
        print("------------------------------\n")

        # Runs the model to get a prediction for that row. 
        prediction = model.predict(df)[0]

        # Returns the predicted price in a user-friendly way
        predicted_price = float(round(prediction, 2))
        return {"predicted_price_gbp": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
