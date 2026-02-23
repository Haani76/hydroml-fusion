cat > README.md << 'EOF'
#  HydroML-Fusion: Multi-Model Ensemble Hydrological Forecasting System

A hydrological modeling framework integrating physics-based models, machine learning, and deep learning with real-time forecasting capabilities and climate scenario analysis.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)


---

##  Project Overview

**HydroML-Fusion** is an advanced hydrological forecasting system that combines:
- **Physics-based modeling** (GR4J)
- **Machine learning** (XGBoost)
- **Deep learning** (LSTM with attention)
- **Multi-model ensemble** with uncertainty quantification
- **Real geospatial data integration** (SRTM DEM, land use)
- **Climate scenario analysis** (CMIP6 projections)
- **REST API** for operational deployment

### Study Area
- **Basin:** Leaf River, Mississippi, USA
- **Area:** 1,950 km²
- **Period:** 2010-2020 (11 years)
- **Data:** Streamflow, precipitation, temperature, DEM, land use

---

##  Model Performance

| Model | NSE | RMSE (mm/day) | Features |
|-------|-----|---------------|----------|
| **LSTM + Geospatial** | **0.928** 🏆 | **0.514** | 13 (climate + terrain + land use) |
| Ensemble (3-model) | 0.915 | 0.558 | Weighted combination |
| LSTM (baseline) | 0.905 | 0.592 | 6 (climate only) |
| XGBoost | 0.737 | 0.974 | 6 |
| GR4J | 0.729 | 1.185 | 4 parameters |

**Key Finding:** Adding geospatial features improved LSTM NSE from 0.905 to 0.928 (+0.023)

---

##  Features

### Hydrological Models
-  **GR4J:** 4-parameter conceptual rainfall-runoff model
-  **XGBoost:** Gradient boosting regression trees
-  **LSTM:** Long Short-Term Memory neural network (PyTorch)
-  **Ensemble:** Optimized multi-model combination

### Geospatial Integration
-  **Real SRTM DEM** (90m resolution via OpenTopography API)
-  **Terrain features:** Elevation, slope, aspect, TWI
-  **Land use classification:** Forest, agriculture, wetland, developed
-  **13 total features** for enhanced predictions

### Climate Scenarios
-  **CMIP6 NetCDF scenarios:** SSP2-4.5, SSP5-8.5
-  **Future projections:** 2041-2060 mid-century
-  **Impact assessment:** -1.8% to -1.3% streamflow change

### Uncertainty Quantification
-  **Monte Carlo Dropout:** 100-sample ensemble
-  **Probabilistic forecasts:** 5th, 25th, 75th, 95th percentiles
-  **Confidence intervals:** 90% prediction bands

### Production Deployment
-  **REST API:** FastAPI with 5 endpoints
-  **Docker:** Multi-stage containerization
-  **Documentation:** Comprehensive guides

---

## 🛠️ Tech Stack

**Programming & Libraries:**
- Python 3.11
- PyTorch 2.10 (Deep Learning)
- XGBoost 2.0 (Machine Learning)
- scikit-learn (Preprocessing, metrics)
- FastAPI (REST API)

**Geospatial:**
- rasterio, GDAL (Raster processing)
- geopandas, shapely (Vector operations)
- xarray, netCDF4 (NetCDF/climate data)

**Data & Analysis:**
- pandas, NumPy (Data processing)
- SciPy (Optimization)
- Matplotlib, Plotly (Visualization)

**Deployment:**
- Docker, docker-compose
- Uvicorn (ASGI server)

---

##  Project Structure
```
hydroml-fusion/
├── data/
│   ├── raw/
│   │   ├── streamflow/           # USGS/synthetic streamflow
│   │   ├── climate/               # Precipitation, temperature, CMIP6
│   │   └── geospatial/            # SRTM DEM, land use rasters
│   ├── processed/                 # Cleaned, feature-engineered data
│   └── forecasts/                 # Model predictions, scenarios
├── models/
│   └── trained/                   # Saved model weights
├── src/
│   ├── data_acquisition/          # Download scripts
│   ├── data_processing/           # ETL pipelines
│   ├── geospatial/                # DEM, land use processing
│   ├── modeling/                  # GR4J, LSTM, XGBoost
│   ├── calibration/               # Parameter optimization
│   ├── forecasting/               # Climate scenarios, uncertainty
│   └── api/                       # FastAPI REST API
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── README.md
├── config.yaml                    # Project configuration
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Haani76/hydroml-fusion.git
cd hydroml-fusion
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Models
```bash
# Train all models
python src/modeling/train_lstm_geospatial.py

# Run climate scenarios
python src/forecasting/run_climate_scenarios.py

# Generate uncertainty forecasts
python src/forecasting/uncertainty_quantification.py
```

### 3. Start REST API
```bash
cd src/api
uvicorn main:app --reload
```

Visit: http://localhost:8000/docs for interactive API documentation

### 4. Docker Deployment
```bash
docker-compose up -d
curl http://localhost:8000/health
```

---

##  Input/Output Parameters

### Inputs

**Climate Variables:**
- Precipitation (mm/day)
- Temperature (°C)
- Potential evapotranspiration (mm/day)

**Terrain Features:**
- Elevation (m)
- Slope (degrees)
- Topographic Wetness Index

**Land Use:**
- Forest coverage (%)
- Agriculture (%)
- Wetland (%)
- Developed areas (%)

### Outputs

**Streamflow:**
- mm/day (basin average)
- cfs (cubic feet per second)

**Uncertainty:**
- Mean prediction
- Standard deviation
- 5th, 25th, 75th, 95th percentiles

---

##  Methodology

### GR4J Model
4-parameter conceptual model with:
- Production store (soil moisture)
- Routing store (groundwater)
- Unit hydrographs (surface runoff)
- Calibrated using differential evolution

### LSTM Deep Learning
- **Architecture:** 2-layer LSTM (64 hidden units)
- **Sequence length:** 30 days
- **Features:** 13 (climate + terrain + land use)
- **Training:** Early stopping, validation monitoring
- **Uncertainty:** Monte Carlo Dropout (100 samples)

### XGBoost Machine Learning
- **Trees:** 200 estimators
- **Depth:** 6
- **Features:** Precipitation, temperature, 7-day averages
- **Importance:** 7-day average temperature (66%)

### Ensemble
- **Method:** Optimized weighted average
- **Weights:** GR4J (1%), XGBoost (19%), LSTM (80%)
- **Optimization:** Minimize RMSE on validation set

---

##  Results

### Model Comparison

**Calibration (2010-2018):**
- GR4J: NSE = 0.678
- XGBoost: NSE = 0.841
- LSTM: NSE = 0.850

**Validation (2019-2020):**
- GR4J: NSE = 0.729
- XGBoost: NSE = 0.737
- LSTM (baseline): NSE = 0.905
- **LSTM + Geospatial: NSE = 0.928** 
- **Ensemble: NSE = 0.915**

### Climate Scenario Impacts (2041-2060)

| Scenario | Temp Change | Precip Change | Streamflow Change |
|----------|-------------|---------------|-------------------|
| SSP2-4.5 | +1.8°C | +5% | **-1.8%** |
| SSP5-8.5 | +2.7°C | +8% | **-1.3%** |

*Despite increased precipitation, higher evapotranspiration leads to reduced streamflow*

---

##  REST API

### Endpoints

**Health & Info:**
- `GET /` - Root health check
- `GET /health` - Detailed status
- `GET /info` - Project details
- `GET /models` - Available models

**Prediction:**
- `POST /predict` - Streamflow forecast

### Example Usage
```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={
        'precipitation_mm': 10.5,
        'temperature_c': 22.0,
        'pet_mm': 3.5
    }
)

print(response.json())
# Output: {'streamflow_mm': 3.11, 'streamflow_cfs': 2481.29, 'model': 'LSTM with geospatial features'}
```

---

##  Skills 


 **Hydrological Model Development**
- Basin-scale rainfall-runoff modeling (GR4J)
- Custom implementation, not just library usage

 **Climate Scenario Integration**
- CMIP6 NetCDF processing
- Future water availability assessment

 **Quantitative Data Analysis**
- Large-scale dataset processing (11 years daily)
- Statistical analysis and validation

 **Calibration & Validation**
- Parameter optimization (differential evolution)
- Sensitivity analysis
- Independent validation period

 **Geospatial Data Processing**
- Real SRTM DEM acquisition and processing
- Terrain feature extraction (slope, aspect, TWI)
- Land use classification
- NetCDF raster operations

 **Programming & Software Development**
- Production-ready Python code
- REST API development
- Docker containerization
- Version control (Git/GitHub)

**Machine Learning & AI**
- Deep learning (LSTM/PyTorch)
- Ensemble methods
- Uncertainty quantification

---

##  Future Enhancements

- [ ] Real-time GFS/ECMWF forecast integration
- [ ] MODIS satellite data (NDVI, ET)
- [ ] Distributed hydrological modeling
- [ ] Data assimilation (Kalman filter)
- [ ] Interactive web dashboard (Streamlit/Dash)
- [ ] Multi-basin regionalization
- [ ] Operational deployment (AWS/GCP)

---

##  References

**GR4J Model:**
- Perrin et al. (2003). *Journal of Hydrology*, 279(1-4), 275-289.

**LSTM in Hydrology:**
- Kratzert et al. (2019). *Water Resources Research*, 55(12), 5364-5377.

**CMIP6:**
- IPCC AR6 (2021). *Climate Change 2021: The Physical Science Basis*

---

##  Contact

**Haani Shafiq Siddiqui**  
GitHub: [@Haani76](https://github.com/Haani76)  
Repository: [hydroml-fusion](https://github.com/Haani76/hydroml-fusion)

---

##  License

MIT License - Free for educational and research purposes

---

**Built with dedication for advancing hydrological forecasting and water resources management** 
