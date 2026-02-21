"""
Multi-Model Ensemble: Combine GR4J, XGBoost, and LSTM
"""

import sys
sys.path.insert(0, 'src/modeling')

import pandas as pd
import numpy as np
from gr4j_model import calculate_nse, calculate_rmse, calculate_bias

print("=" * 60)
print("MULTI-MODEL ENSEMBLE")
print("=" * 60)

# Load all model predictions
gr4j = pd.read_csv("data/processed/gr4j_validation_results.csv", parse_dates=['date'])
xgb_results = pd.read_csv("data/processed/xgboost_validation_results.csv", parse_dates=['date'])
lstm = pd.read_csv("data/processed/lstm_validation_results.csv", parse_dates=['date'])

print(f"\n✓ Loaded predictions from all 3 models")
print(f"  Records: {len(gr4j)}")

# Align datasets (LSTM has shorter records due to sequence length)
# Use only dates where all models have predictions
common_dates = set(gr4j['date']) & set(xgb_results['date']) & set(lstm['date'])
common_dates = sorted(list(common_dates))

print(f"  Common dates: {len(common_dates)}")

# Filter to common dates
gr4j_filtered = gr4j[gr4j['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
xgb_filtered = xgb_results[xgb_results['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)
lstm_filtered = lstm[lstm['date'].isin(common_dates)].sort_values('date').reset_index(drop=True)

# Extract predictions
obs = gr4j_filtered['observed'].values
pred_gr4j = gr4j_filtered['simulated'].values
pred_xgb = xgb_filtered['predicted'].values
pred_lstm = lstm_filtered['predicted'].values

print(f"\n✓ Aligned predictions")

# Calculate individual model NSE on common dates
nse_gr4j = calculate_nse(obs, pred_gr4j)
nse_xgb = calculate_nse(obs, pred_xgb)
nse_lstm = calculate_nse(obs, pred_lstm)

print(f"\nIndividual Model Performance (on common dates):")
print(f"  GR4J:    NSE = {nse_gr4j:.3f}")
print(f"  XGBoost: NSE = {nse_xgb:.3f}")
print(f"  LSTM:    NSE = {nse_lstm:.3f}")

# Method 1: Simple Average
ensemble_avg = (pred_gr4j + pred_xgb + pred_lstm) / 3
nse_avg = calculate_nse(obs, ensemble_avg)

print(f"\nEnsemble Method 1: Simple Average")
print(f"  NSE = {nse_avg:.3f}")

# Method 2: Weighted by NSE
total_nse = nse_gr4j + nse_xgb + nse_lstm
w_gr4j = nse_gr4j / total_nse
w_xgb = nse_xgb / total_nse
w_lstm = nse_lstm / total_nse

ensemble_weighted = w_gr4j * pred_gr4j + w_xgb * pred_xgb + w_lstm * pred_lstm
nse_weighted = calculate_nse(obs, ensemble_weighted)

print(f"\nEnsemble Method 2: NSE-Weighted")
print(f"  Weights: GR4J={w_gr4j:.2f}, XGBoost={w_xgb:.2f}, LSTM={w_lstm:.2f}")
print(f"  NSE = {nse_weighted:.3f}")

# Method 3: Optimal weights (minimize error)
from scipy.optimize import minimize

def ensemble_error(weights):
    weights = weights / weights.sum()  # Normalize
    ensemble_pred = (weights[0] * pred_gr4j + 
                     weights[1] * pred_xgb + 
                     weights[2] * pred_lstm)
    return -calculate_nse(obs, ensemble_pred)  # Minimize negative NSE

result = minimize(
    ensemble_error,
    x0=[1/3, 1/3, 1/3],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
)

optimal_weights = result.x
ensemble_optimal = (optimal_weights[0] * pred_gr4j + 
                   optimal_weights[1] * pred_xgb + 
                   optimal_weights[2] * pred_lstm)
nse_optimal = calculate_nse(obs, ensemble_optimal)

print(f"\nEnsemble Method 3: Optimized Weights")
print(f"  Weights: GR4J={optimal_weights[0]:.2f}, XGBoost={optimal_weights[1]:.2f}, LSTM={optimal_weights[2]:.2f}")
print(f"  NSE = {nse_optimal:.3f}")

# Use best ensemble
best_method = max([
    ('Simple Average', nse_avg, ensemble_avg),
    ('NSE-Weighted', nse_weighted, ensemble_weighted),
    ('Optimized', nse_optimal, ensemble_optimal)
], key=lambda x: x[1])

print(f"\n" + "=" * 60)
print(f"BEST ENSEMBLE: {best_method[0]}")
print(f"NSE = {best_method[1]:.3f}")
print("=" * 60)

# Calculate final metrics
final_pred = best_method[2]
rmse = calculate_rmse(obs, final_pred)
bias = calculate_bias(obs, final_pred)

print(f"\nFinal Ensemble Performance:")
print(f"  NSE:  {best_method[1]:.3f}")
print(f"  RMSE: {rmse:.3f} mm/day")
print(f"  Bias: {bias:.3f} mm/day")

# Save ensemble results
results = pd.DataFrame({
    'date': gr4j_filtered['date'],
    'observed': obs,
    'gr4j': pred_gr4j,
    'xgboost': pred_xgb,
    'lstm': pred_lstm,
    'ensemble': final_pred
})
results.to_csv("data/processed/ensemble_results.csv", index=False)

print(f"\n✓ Saved ensemble results")
print("\n" + "=" * 60)
print("ENSEMBLE COMPLETE")
print("=" * 60)