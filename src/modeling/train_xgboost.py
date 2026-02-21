"""
Train XGBoost model for streamflow prediction
"""

import sys
sys.path.insert(0, 'src/modeling')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle

from gr4j_model import calculate_nse, calculate_rmse, calculate_bias

print("=" * 60)
print("TRAINING XGBOOST MODEL")
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

print(f"\n✓ Prepared features")
print(f"  Features: {feature_cols}")

# Train XGBoost
print("\n" + "-" * 60)
print("Training XGBoost...")
print("-" * 60)

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=20
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

# Make predictions
print("\nEvaluating on validation data...")

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

# Calculate metrics
nse_train = calculate_nse(y_train, y_pred_train)
nse_val = calculate_nse(y_val, y_pred_val)
rmse_val = calculate_rmse(y_val, y_pred_val)
bias_val = calculate_bias(y_val, y_pred_val)

print(f"\nTraining Performance:")
print(f"  NSE: {nse_train:.3f}")

print(f"\nValidation Performance:")
print(f"  NSE:  {nse_val:.3f}")
print(f"  RMSE: {rmse_val:.3f} mm/day")
print(f"  Bias: {bias_val:.3f} mm/day")

# Feature importance
print(f"\nFeature Importance:")
importance = model.feature_importances_
for feat, imp in zip(feature_cols, importance):
    print(f"  {feat:20s}: {imp:.3f}")

# Save model
model.save_model('models/trained/xgboost_model.json')
print(f"\n✓ Saved model to: models/trained/xgboost_model.json")

# Save validation results
results = pd.DataFrame({
    'date': df_val['date'],
    'observed': y_val,
    'predicted': y_pred_val
})
results.to_csv("data/processed/xgboost_validation_results.csv", index=False)
print(f"✓ Saved validation results")

print("\n" + "=" * 60)
print("XGBOOST MODEL COMPLETE")
print("=" * 60)