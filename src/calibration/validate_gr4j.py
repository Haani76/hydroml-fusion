"""
Validate GR4J model on independent dataset
"""

import sys
sys.path.insert(0, 'src/modeling')

import pandas as pd
import numpy as np
from gr4j_model import GR4J, calculate_nse, calculate_rmse, calculate_bias

print("=" * 60)
print("VALIDATING GR4J MODEL")
print("=" * 60)

# Load parameters
params = pd.read_csv("data/processed/gr4j_calibrated_params.csv")
X1 = params[params['parameter'] == 'X1']['value'].values[0]
X2 = params[params['parameter'] == 'X2']['value'].values[0]
X3 = params[params['parameter'] == 'X3']['value'].values[0]
X4 = params[params['parameter'] == 'X4']['value'].values[0]

print(f"\nUsing calibrated parameters:")
print(f"  X1 = {X1:.2f} mm")
print(f"  X2 = {X2:.2f} mm")
print(f"  X3 = {X3:.2f} mm")
print(f"  X4 = {X4:.2f} days")

# Load validation data
df = pd.read_csv("data/processed/validation_data.csv", parse_dates=['date'])

print(f"\nValidation period: {df['date'].min()} to {df['date'].max()}")
print(f"Records: {len(df)}")

# Run model
precip = df['precipitation_mm'].values
pet = df['pet_mm'].values
obs_flow = df['streamflow_mm'].values

model = GR4J(X1=X1, X2=X2, X3=X3, X4=X4)
sim_flow = model.run(precip, pet)

# Calculate metrics
nse = calculate_nse(obs_flow, sim_flow)
rmse = calculate_rmse(obs_flow, sim_flow)
bias = calculate_bias(obs_flow, sim_flow)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print(f"\nPerformance Metrics:")
print(f"  NSE:  {nse:.3f}")
print(f"  RMSE: {rmse:.3f} mm/day")
print(f"  Bias: {bias:.3f} mm/day")

# Save results
results = pd.DataFrame({
    'date': df['date'],
    'observed': obs_flow,
    'simulated': sim_flow
})
results.to_csv("data/processed/gr4j_validation_results.csv", index=False)

print(f"\n✓ Saved validation results")
print("\n" + "=" * 60)
print("GR4J MODEL COMPLETE")
print("=" * 60)