"""
REST API for HydroML-Fusion
FastAPI implementation for model predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '../modeling')

from lstm_model import LSTMModel

# Initialize FastAPI
app = FastAPI(
    title="HydroML-Fusion API",
    description="Hydrological forecasting with multi-model ensemble",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class PredictionRequest(BaseModel):
    precipitation_mm: float
    temperature_c: float
    pet_mm: Optional[float] = None
    elevation_m: float = 91.0
    slope_deg: float = 1.28
    twi: float = 8.41
    forest_pct: float = 65.35
    agriculture_pct: float = 14.66
    wetland_pct: float = 1.63
    developed_pct: float = 3.22

class PredictionResponse(BaseModel):
    streamflow_mm: float
    streamflow_cfs: float
    model: str
    uncertainty_mm: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict

# Global variables (load models once)
lstm_model = None
scaler_X = None
scaler_y = None

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global lstm_model, scaler_X, scaler_y
    
    # Load LSTM
    lstm_model = LSTMModel(input_size=13, hidden_size=64, num_layers=2, dropout=0.2)
    lstm_model.load_state_dict(torch.load('../../models/trained/lstm_geospatial_model.pth'))
    lstm_model.eval()
    
    # Prepare scalers
    df_train = pd.read_csv("../../data/processed/calibration_data_with_geospatial.csv")
    
    feature_cols = [
        'precipitation_mm', 'temperature_c', 'pet_mm', 
        'precip_7day', 'temp_7day', 'pet_7day',
        'elevation_m', 'slope_deg', 'twi',
        'forest_pct', 'agriculture_pct', 'wetland_pct', 'developed_pct'
    ]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['streamflow_mm'].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(X_train)
    scaler_y.fit(y_train.reshape(-1, 1))
    
    print("✓ Models loaded successfully")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "operational",
        "models_loaded": {
            "lstm": lstm_model is not None,
            "scalers": scaler_X is not None
        }
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "models": {
            "lstm_geospatial": "loaded" if lstm_model is not None else "not_loaded"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make streamflow prediction
    
    Parameters:
    - precipitation_mm: Daily precipitation (mm)
    - temperature_c: Daily temperature (°C)
    - pet_mm: Potential evapotranspiration (optional, calculated if not provided)
    - elevation_m, slope_deg, twi: Terrain features (defaults provided)
    - forest_pct, agriculture_pct, wetland_pct, developed_pct: Land use (defaults provided)
    
    Returns:
    - streamflow_mm: Predicted streamflow (mm/day)
    - streamflow_cfs: Predicted streamflow (cubic feet per second)
    """
    
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate PET if not provided
    if request.pet_mm is None:
        temp = request.temperature_c
        pet = 0.0023 * (temp + 17.8) * np.sqrt(abs(temp - (-5))) * 2.5
        pet = max(pet, 0)
    else:
        pet = request.pet_mm
    
    # Prepare features (using single values, assuming historical context)
    features = np.array([[
        request.precipitation_mm,
        request.temperature_c,
        pet,
        request.precipitation_mm,  # precip_7day (simplified)
        request.temperature_c,      # temp_7day (simplified)
        pet,                        # pet_7day (simplified)
        request.elevation_m,
        request.slope_deg,
        request.twi,
        request.forest_pct,
        request.agriculture_pct,
        request.wetland_pct,
        request.developed_pct
    ]])
    
    # Scale
    features_scaled = scaler_X.transform(features)
    
    # Create sequence (repeat for sequence length)
    sequence_length = 30
    sequence = np.tile(features_scaled, (sequence_length, 1))
    
    # Predict
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
    
    with torch.no_grad():
        prediction_scaled = lstm_model(sequence_tensor)
    
    # Inverse transform
    prediction_mm = scaler_y.inverse_transform(prediction_scaled.numpy())[0, 0]
    
    # Convert to cfs (assuming basin area 1950 km²)
    basin_area_km2 = 1950
    mm_to_cfs = (basin_area_km2 * 1e6) / 86400 / 0.0283168 / 1000
    prediction_cfs = prediction_mm * mm_to_cfs
    
    return {
        "streamflow_mm": float(prediction_mm),
        "streamflow_cfs": float(prediction_cfs),
        "model": "LSTM with geospatial features"
    }

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": [
            {
                "name": "LSTM Geospatial",
                "type": "Deep Learning",
                "features": 13,
                "validation_nse": 0.928
            }
        ]
    }

@app.get("/info")
async def info():
    """API information"""
    return {
        "project": "HydroML-Fusion",
        "description": "Multi-model ensemble hydrological forecasting system",
        "basin": "Leaf River, Mississippi",
        "models": ["GR4J", "XGBoost", "LSTM", "Ensemble"],
        "features": {
            "climate": ["precipitation", "temperature", "PET"],
            "terrain": ["elevation", "slope", "TWI"],
            "land_use": ["forest", "agriculture", "wetland", "developed"]
        },
        "capabilities": [
            "Streamflow prediction",
            "Climate scenario analysis",
            "Uncertainty quantification"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)