"""
Uncertainty Quantification using Monte Carlo Dropout
Generate probabilistic streamflow forecasts
"""

import sys
sys.path.insert(0, 'src/modeling')

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from lstm_model import LSTMModel

print("=" * 60)
print("UNCERTAINTY QUANTIFICATION")
print("=" * 60)

print("\nMethod: Monte Carlo Dropout")
print("Technique: Multiple forward passes with dropout enabled")

# Prepare data
df_train = pd.read_csv("data/processed/calibration_data_with_geospatial.csv", parse_dates=['date'])
df_val = pd.read_csv("data/processed/validation_data_with_geospatial.csv", parse_dates=['date'])

feature_cols = [
    'precipitation_mm', 'temperature_c', 'pet_mm', 
    'precip_7day', 'temp_7day', 'pet_7day',
    'elevation_m', 'slope_deg', 'twi',
    'forest_pct', 'agriculture_pct', 'wetland_pct', 'developed_pct'
]

X_train = df_train[feature_cols].values
y_train = df_train['streamflow_mm'].values
X_val = df_val[feature_cols].values
y_val = df_val['streamflow_mm'].values

# Scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

X_val_scaled = scaler_X.transform(X_val)

print(f"\n✓ Data prepared")

# Load model
model = LSTMModel(input_size=13, hidden_size=64, num_layers=2, dropout=0.2)
model.load_state_dict(torch.load('models/trained/lstm_geospatial_model.pth'))

print(f"✓ Loaded LSTM model")

# Monte Carlo Dropout predictions
sequence_length = 30
n_samples = 100  # Number of stochastic forward passes

print(f"\nGenerating {n_samples} probabilistic predictions...")
print("(This takes ~1 minute)")

# Store all predictions
all_predictions = []

for sample in range(n_samples):
    if (sample + 1) % 20 == 0:
        print(f"  Sample {sample+1}/{n_samples}...")
    
    # Enable dropout during inference
    model.train()  # This keeps dropout active
    
    predictions_scaled = []
    
    for i in range(len(X_val_scaled) - sequence_length):
        seq = X_val_scaled[i:i+sequence_length]
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
        
        with torch.no_grad():
            pred = model(seq_tensor)
        
        predictions_scaled.append(pred.item())
    
    # Inverse transform
    predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    all_predictions.append(predictions)

# Convert to array (n_samples x n_timesteps)
all_predictions = np.array(all_predictions)

print(f"\n✓ Generated {n_samples} predictions")

# Calculate statistics
mean_pred = all_predictions.mean(axis=0)
std_pred = all_predictions.std(axis=0)
p05 = np.percentile(all_predictions, 5, axis=0)   # 5th percentile
p25 = np.percentile(all_predictions, 25, axis=0)  # 25th percentile
p75 = np.percentile(all_predictions, 75, axis=0)  # 75th percentile
p95 = np.percentile(all_predictions, 95, axis=0)  # 95th percentile

print(f"\nUncertainty statistics:")
print(f"  Mean prediction: {mean_pred.mean():.2f} mm/day")
print(f"  Mean std dev: {std_pred.mean():.2f} mm/day")
print(f"  90% confidence interval width: {(p95 - p05).mean():.2f} mm/day")

# Save results
results = pd.DataFrame({
    'date': df_val['date'].iloc[sequence_length:].values,
    'observed': y_val[sequence_length:],
    'mean_prediction': mean_pred,
    'std_prediction': std_pred,
    'p05': p05,
    'p25': p25,
    'p75': p75,
    'p95': p95
})

results.to_csv("data/forecasts/probabilistic_forecasts.csv", index=False)

print(f"\n✓ Saved: data/forecasts/probabilistic_forecasts.csv")

# Calculate prediction interval coverage
within_90 = np.sum((y_val[sequence_length:] >= p05) & (y_val[sequence_length:] <= p95))
coverage_90 = (within_90 / len(p05)) * 100

print(f"\nPrediction interval coverage:")
print(f"  90% interval: {coverage_90:.1f}% (target: 90%)")

print("\n" + "=" * 60)
print("UNCERTAINTY QUANTIFICATION COMPLETE")
print("=" * 60)

print("\nOutputs:")
print("  ✓ Mean prediction")
print("  ✓ Standard deviation")
print("  ✓ 5th, 25th, 75th, 95th percentiles")
print("  ✓ 90% confidence intervals")