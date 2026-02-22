"""
Train LSTM model with geospatial features
Compare performance: with vs without geospatial data
"""

import sys
sys.path.insert(0, 'src/modeling')

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pickle

from lstm_model import LSTMModel, LSTMTrainer, StreamflowDataset
from torch.utils.data import DataLoader
from gr4j_model import calculate_nse, calculate_rmse

print("=" * 60)
print("LSTM WITH GEOSPATIAL FEATURES")
print("=" * 60)

# Load data WITH geospatial features
df_train = pd.read_csv("data/processed/calibration_data_with_geospatial.csv", parse_dates=['date'])
df_val = pd.read_csv("data/processed/validation_data_with_geospatial.csv", parse_dates=['date'])

print(f"\nTraining data: {len(df_train)} records")
print(f"Validation data: {len(df_val)} records")

# Extended feature set
feature_cols = [
    # Climate (original 6)
    'precipitation_mm', 'temperature_c', 'pet_mm', 
    'precip_7day', 'temp_7day', 'pet_7day',
    # Terrain (new 3)
    'elevation_m', 'slope_deg', 'twi',
    # Land use (new 4)
    'forest_pct', 'agriculture_pct', 'wetland_pct', 'developed_pct'
]
target_col = 'streamflow_mm'

print(f"\nFeatures ({len(feature_cols)}):")
print(f"  Climate: 6")
print(f"  Terrain: 3")
print(f"  Land use: 4")

# Prepare data
X_train = df_train[feature_cols].values
y_train = df_train[target_col].values

X_val = df_val[feature_cols].values
y_val = df_val[target_col].values

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

print(f"\n✓ Data normalized")

# Create datasets
sequence_length = 30

train_dataset = StreamflowDataset(X_train_scaled, y_train_scaled, sequence_length)
val_dataset = StreamflowDataset(X_val_scaled, y_val_scaled, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"✓ Created datasets")

# Create model (13 input features now!)
input_size = len(feature_cols)
model = LSTMModel(
    input_size=input_size,
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)

print(f"\n✓ Created LSTM model")
print(f"  Input features: {input_size}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
print("\n" + "-" * 60)
print("Training with geospatial features...")
print("-" * 60)

trainer = LSTMTrainer(model)
train_losses, val_losses = trainer.train(
    train_loader, 
    val_loader, 
    epochs=50, 
    verbose=True
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

# Evaluate
print("\nEvaluating...")

val_sequences = []
for i in range(len(X_val_scaled) - sequence_length):
    val_sequences.append(X_val_scaled[i:i+sequence_length])

val_sequences = np.array(val_sequences)
predictions_scaled = trainer.predict(val_sequences)

# Inverse transform
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
y_val_actual = y_val[sequence_length:]

# Calculate metrics
nse = calculate_nse(y_val_actual, predictions)
rmse = calculate_rmse(y_val_actual, predictions)

print(f"\nPerformance WITH Geospatial Features:")
print(f"  NSE:  {nse:.3f}")
print(f"  RMSE: {rmse:.3f} mm/day")

# Compare with original LSTM (without geospatial)
original_results = pd.read_csv("data/processed/lstm_validation_results.csv", parse_dates=['date'])
original_nse = calculate_nse(
    original_results['observed'].values, 
    original_results['predicted'].values
)

print(f"\nPerformance WITHOUT Geospatial Features:")
print(f"  NSE:  {original_nse:.3f}")

print(f"\n" + "=" * 60)
print(f"IMPROVEMENT: {nse - original_nse:+.3f} NSE")
print("=" * 60)

# Save
torch.save(model.state_dict(), 'models/trained/lstm_geospatial_model.pth')
print(f"\n✓ Saved model: models/trained/lstm_geospatial_model.pth")

results = pd.DataFrame({
    'date': df_val['date'].iloc[sequence_length:].values,
    'observed': y_val_actual,
    'predicted': predictions
})
results.to_csv("data/processed/lstm_geospatial_results.csv", index=False)
print(f"✓ Saved results: data/processed/lstm_geospatial_results.csv")

print("\n" + "=" * 60)
print("GEOSPATIAL MODEL COMPLETE")
print("=" * 60)