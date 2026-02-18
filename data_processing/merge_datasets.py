"""
Merge streamflow and climate data
"""

import pandas as pd
from pathlib import Path
import os

print("=" * 60)
print("MERGING DATASETS")
print("=" * 60)

# Load streamflow
streamflow = pd.read_csv("data/raw/streamflow/02472000_streamflow.csv", parse_dates=['date'])
print(f"\n✓ Loaded streamflow: {len(streamflow)} records")

# Load climate
climate = pd.read_csv("data/raw/climate/leaf_river_climate.csv", parse_dates=['date'])
print(f"✓ Loaded climate: {len(climate)} records")

# Merge on date
data = streamflow.merge(climate, on='date', how='inner')

# Convert streamflow to mm/day
basin_area_km2 = 1950
cfs_to_mm = 86400 / (basin_area_km2 * 1e6) * 0.0283168 * 1000
data['streamflow_mm'] = data['streamflow_cfs'] * cfs_to_mm

# Calculate additional features
data['precip_7day'] = data['precipitation_mm'].rolling(7, min_periods=1).mean()
data['temp_7day'] = data['temperature_c'].rolling(7, min_periods=1).mean()
data['pet_7day'] = data['pet_mm'].rolling(7, min_periods=1).mean()

# Remove any NaN
data = data.dropna()

# Split into calibration and validation
split_date = '2019-01-01'
data_cal = data[data['date'] < split_date].copy()
data_val = data[data['date'] >= split_date].copy()

# Save (folder already exists, just save files)
data.to_csv("data/processed/complete_data.csv", index=False)
data_cal.to_csv("data/processed/calibration_data.csv", index=False)
data_val.to_csv("data/processed/validation_data.csv", index=False)

print(f"\n✓ Merged and processed data")
print(f"  Total records: {len(data)}")
print(f"  Calibration: {len(data_cal)} (2010-2018)")
print(f"  Validation: {len(data_val)} (2019-2020)")
print(f"\n✓ Saved to data/processed/")

print("\n" + "=" * 60)
print("DATA PROCESSING COMPLETE")
print("=" * 60)