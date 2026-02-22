"""
Integrate geospatial features into hydrological dataset
Add: elevation, slope, TWI, land use percentages
"""

import pandas as pd
from pathlib import Path

print("=" * 60)
print("INTEGRATING GEOSPATIAL FEATURES")
print("=" * 60)

# Load existing hydrological data
df = pd.read_csv("data/processed/complete_data.csv", parse_dates=['date'])

print(f"\n✓ Loaded hydrological data")
print(f"  Records: {len(df)}")
print(f"  Current features: {len(df.columns)} columns")

# Load terrain features
terrain = pd.read_csv("data/processed/terrain_features.csv")
print(f"\n✓ Loaded terrain features:")
for _, row in terrain.iterrows():
    print(f"  {row['feature']:20s}: {row['value']:.2f}")

# Load land use features
landuse = pd.read_csv("data/processed/landuse_features.csv")
print(f"\n✓ Loaded land use features:")
for _, row in landuse.iterrows():
    print(f"  {row['feature']:20s}: {row['value']:.2f}%")

# Add terrain features (constant for all time steps)
for _, row in terrain.iterrows():
    df[row['feature']] = row['value']

# Add land use features
for _, row in landuse.iterrows():
    df[row['feature']] = row['value']

print(f"\n✓ Added geospatial features")
print(f"  New total: {len(df.columns)} columns")

# Verify new features
new_features = ['elevation_m', 'slope_deg', 'twi', 
                'forest_pct', 'agriculture_pct', 'wetland_pct', 'developed_pct']

print(f"\nGeospatial features added:")
for feat in new_features:
    if feat in df.columns:
        print(f"  ✓ {feat}")
    else:
        print(f"  ✗ {feat} MISSING")

# Save enhanced dataset
df.to_csv("data/processed/complete_data_with_geospatial.csv", index=False)

print(f"\n✓ Saved enhanced dataset:")
print(f"  data/processed/complete_data_with_geospatial.csv")

# Split into calibration and validation
split_date = '2019-01-01'
df_cal = df[df['date'] < split_date].copy()
df_val = df[df['date'] >= split_date].copy()

df_cal.to_csv("data/processed/calibration_data_with_geospatial.csv", index=False)
df_val.to_csv("data/processed/validation_data_with_geospatial.csv", index=False)

print(f"\n✓ Saved calibration/validation splits")
print(f"  Calibration: {len(df_cal)} records")
print(f"  Validation: {len(df_val)} records")

# Show final feature list
print(f"\n" + "=" * 60)
print("FINAL FEATURE SET")
print("=" * 60)

feature_groups = {
    'Climate': ['precipitation_mm', 'temperature_c', 'pet_mm'],
    'Temporal': ['precip_7day', 'temp_7day', 'pet_7day'],
    'Terrain': ['elevation_m', 'slope_deg', 'twi'],
    'Land Use': ['forest_pct', 'agriculture_pct', 'wetland_pct', 'developed_pct'],
    'Target': ['streamflow_mm', 'streamflow_cfs']
}

for group, features in feature_groups.items():
    print(f"\n{group}:")
    for feat in features:
        if feat in df.columns:
            print(f"  ✓ {feat}")

print("\n" + "=" * 60)
print("GEOSPATIAL INTEGRATION COMPLETE")
print("=" * 60)