"""
Run hydrological models with CMIP6 future climate scenarios
"""

import sys
sys.path.insert(0, 'src/modeling')

import pandas as pd
import numpy as np
import xarray as xr
import torch
from sklearn.preprocessing import StandardScaler

from lstm_model import LSTMModel
from gr4j_model import calculate_nse

print("=" * 60)
print("FUTURE CLIMATE IMPACT ASSESSMENT")
print("=" * 60)

# We need to create a new scaler based on the geospatial training data
print("\nPreparing scalers from geospatial training data...")

df_train = pd.read_csv("data/processed/calibration_data_with_geospatial.csv", parse_dates=['date'])

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

print("✓ Scalers prepared")

# Load climate scenarios
scenarios = ['ssp2_4.5', 'ssp5_8.5']

results_summary = []

for scenario in scenarios:
    print(f"\n" + "=" * 60)
    print(f"SCENARIO: {scenario.upper().replace('_', '-')}")
    print("=" * 60)
    
    # Load NetCDF
    nc_file = f"data/raw/climate/cmip6_{scenario}.nc"
    ds = xr.open_dataset(nc_file)
    
    print(f"\n✓ Loaded NetCDF scenario")
    print(f"  Period: {ds.attrs['projection_period']}")
    print(f"  Mean temp: {ds['tas'].mean().values:.1f}°C")
    print(f"  Mean precip: {ds['pr'].mean().values:.2f} mm/day")
    
    # Load CSV
    df_future = pd.read_csv(f"data/processed/climate_scenario_{scenario}.csv", parse_dates=['date'])
    
    print(f"\nRunning LSTM model...")
    
    # Load model
    model = LSTMModel(input_size=13, hidden_size=64, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load('models/trained/lstm_geospatial_model.pth'))
    model.eval()
    
    # Prepare features
    X_future = df_future[feature_cols].values
    X_future_scaled = scaler_X.transform(X_future)
    
    # Make predictions
    sequence_length = 30
    predictions_scaled = []
    
    for i in range(len(X_future_scaled) - sequence_length):
        seq = X_future_scaled[i:i+sequence_length]
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
        
        with torch.no_grad():
            pred = model(seq_tensor)
        
        predictions_scaled.append(pred.item())
    
    # Inverse transform
    predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    
    # Statistics
    future_mean_flow = predictions.mean()
    future_annual_flow = future_mean_flow * 365
    
    print(f"\n  Future streamflow:")
    print(f"    Mean: {future_mean_flow:.2f} mm/day")
    print(f"    Annual: {future_annual_flow:.0f} mm/year")
    
    # Compare with baseline
    df_hist = pd.read_csv("data/processed/complete_data_with_geospatial.csv")
    hist_mean_flow = df_hist['streamflow_mm'].mean()
    
    change_pct = ((future_mean_flow - hist_mean_flow) / hist_mean_flow) * 100
    
    print(f"\n  vs Historical:")
    print(f"    Historical: {hist_mean_flow:.2f} mm/day")
    print(f"    Change: {change_pct:+.1f}%")
    
    # Save
    results_df = pd.DataFrame({
        'date': df_future['date'].iloc[sequence_length:].values,
        'precipitation_mm': df_future['precipitation_mm'].iloc[sequence_length:].values,
        'temperature_c': df_future['temperature_c'].iloc[sequence_length:].values,
        'predicted_streamflow_mm': predictions
    })
    
    output_file = f"data/forecasts/scenario_{scenario}_projections.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n  ✓ Saved: {output_file}")
    
    results_summary.append({
        'scenario': scenario.upper().replace('_', '-'),
        'mean_streamflow_mm': future_mean_flow,
        'change_pct': change_pct
    })
    
    ds.close()

# Summary
print("\n" + "=" * 60)
print("CLIMATE IMPACT SUMMARY")
print("=" * 60)

summary_df = pd.DataFrame(results_summary)
print(f"\n{summary_df.to_string(index=False)}")

summary_df.to_csv("data/forecasts/climate_impact_summary.csv", index=False)

print("\n✓ Saved: data/forecasts/climate_impact_summary.csv")
print("\n" + "=" * 60)
print("CLIMATE SCENARIOS COMPLETE")
print("=" * 60)