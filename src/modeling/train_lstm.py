"""
Train LSTM model for streamflow prediction
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
print("TRAINING LSTM DEEP LEARNING MODEL")
print("=" * 60)

# Load data
df_train = pd.read_csv("data/processed/calibration_data.csv", parse_dates=['date'])
df_val = pd.read_csv("data/processed/validation_data.csv", parse_dates=['date'])

print(f"\nTraining data: {len(df_train)} records")
print(f"Validation data: {len(df_val)} records")

# Select features
feature_cols = ['precipitation_mm', 'temperature_c', 'pet_mm', 'precip_7day', 'temp_7day', 'pet_7day']
target_col = 'streamflow_mm'

# Prepare data
X_train = df_train[feature_cols].values
y_train = df_train[target_col].values

X_val = df_val[feature_cols].values
y_val = df_val[target_col].values

# Normalize features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

print(f"\n✓ Data normalized")
print(f"  Features: {len(feature_cols)}")
print(f"  Sequence length: 30 days")

# Create datasets
sequence_length = 30

train_dataset = StreamflowDataset(X_train_scaled, y_train_scaled, sequence_length)
val_dataset = StreamflowDataset(X_val_scaled, y_val_scaled, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"\n✓ Created datasets")
print(f"  Training batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")

# Create model
input_size = len(feature_cols)
model = LSTMModel(
    input_size=input_size,
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)

print(f"\n✓ Created LSTM model")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train model
print("\n" + "-" * 60)
print("Training (this will take 2-3 minutes)...")
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

# Evaluate on validation set
print("\nEvaluating on validation data...")

# Prepare validation sequences
val_sequences = []
for i in range(len(X_val_scaled) - sequence_length):
    val_sequences.append(X_val_scaled[i:i+sequence_length])

val_sequences = np.array(val_sequences)
predictions_scaled = trainer.predict(val_sequences)

# Inverse transform predictions
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
y_val_actual = y_val[sequence_length:]

# Calculate metrics
nse = calculate_nse(y_val_actual, predictions)
rmse = calculate_rmse(y_val_actual, predictions)

print(f"\nValidation Performance:")
print(f"  NSE:  {nse:.3f}")
print(f"  RMSE: {rmse:.3f} mm/day")

# Save model
torch.save(model.state_dict(), 'models/trained/lstm_model.pth')
print(f"\n✓ Saved model to: models/trained/lstm_model.pth")

# Save scalers
with open('models/trained/lstm_scalers.pkl', 'wb') as f:
    pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
print(f"✓ Saved scalers")

# Save validation results
results = pd.DataFrame({
    'date': df_val['date'].iloc[sequence_length:].values,
    'observed': y_val_actual,
    'predicted': predictions
})
results.to_csv("data/processed/lstm_validation_results.csv", index=False)
print(f"✓ Saved validation results")

print("\n" + "=" * 60)
print("LSTM MODEL COMPLETE")
print("=" * 60)