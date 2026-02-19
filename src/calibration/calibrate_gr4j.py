"""
Calibrate GR4J model parameters using differential evolution
"""

import sys
sys.path.insert(0, 'src/modeling')

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from gr4j_model import GR4J, calculate_nse

print("=" * 60)
print("CALIBRATING GR4J MODEL")
print("=" * 60)

# Load calibration data
df = pd.read_csv("data/processed/calibration_data.csv", parse_dates=['date'])

print(f"\nCalibration period: {df['date'].min()} to {df['date'].max()}")
print(f"Records: {len(df)}")

# Prepare inputs
precip = df['precipitation_mm'].values
pet = df['pet_mm'].values
obs_flow = df['streamflow_mm'].values

print(f"\nObserved flow statistics:")
print(f"  Mean: {obs_flow.mean():.2f} mm/day")
print(f"  Min: {obs_flow.min():.2f} mm/day")
print(f"  Max: {obs_flow.max():.2f} mm/day")

# Objective function
def objective(params):
    X1, X2, X3, X4 = params
    
    model = GR4J(X1=X1, X2=X2, X3=X3, X4=X4)
    sim_flow = model.run(precip, pet)
    
    nse = calculate_nse(obs_flow, sim_flow)
    
    return -nse  # Minimize negative NSE

# Parameter bounds
bounds = [
    (100, 1200),   # X1
    (-5, 3),       # X2
    (20, 300),     # X3
    (1.1, 2.9)     # X4
]

print("\n" + "-" * 60)
print("Starting optimization (this will take 2-3 minutes)...")
print("-" * 60)

# Run optimization
result = differential_evolution(
    objective,
    bounds,
    maxiter=50,
    popsize=10,
    seed=42,
    disp=True,
    workers=1
)

# Extract results
X1_opt, X2_opt, X3_opt, X4_opt = result.x
nse_opt = -result.fun

print("\n" + "=" * 60)
print("CALIBRATION COMPLETE")
print("=" * 60)

print(f"\nOptimized Parameters:")
print(f"  X1 (Production store):  {X1_opt:.2f} mm")
print(f"  X2 (Groundwater exch.): {X2_opt:.2f} mm")
print(f"  X3 (Routing store):     {X3_opt:.2f} mm")
print(f"  X4 (Time base):         {X4_opt:.2f} days")

print(f"\nPerformance:")
print(f"  NSE: {nse_opt:.3f}")

# Test model with optimized parameters
model = GR4J(X1=X1_opt, X2=X2_opt, X3=X3_opt, X4=X4_opt)
sim_flow = model.run(precip, pet)

# Save parameters
params_df = pd.DataFrame({
    'parameter': ['X1', 'X2', 'X3', 'X4'],
    'value': [X1_opt, X2_opt, X3_opt, X4_opt]
})
params_df.to_csv("data/processed/gr4j_calibrated_params.csv", index=False)

print(f"\n✓ Saved parameters to: data/processed/gr4j_calibrated_params.csv")

# Save calibration results
results_df = pd.DataFrame({
    'date': df['date'],
    'observed': obs_flow,
    'simulated': sim_flow
})
results_df.to_csv("data/processed/gr4j_calibration_results.csv", index=False)

print(f"✓ Saved results to: data/processed/gr4j_calibration_results.csv")
print("\n" + "=" * 60)
