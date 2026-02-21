"""
Process DEM to extract terrain features
Calculate: slope, aspect, elevation statistics
"""

import rasterio
import numpy as np
from pathlib import Path

print("=" * 60)
print("PROCESSING DEM - EXTRACTING TERRAIN FEATURES")
print("=" * 60)

# Load DEM
dem_file = "data/raw/geospatial/leaf_river_dem.tif"

with rasterio.open(dem_file) as src:
    elevation = src.read(1)
    profile = src.profile
    transform = src.transform
    
    # Get pixel size (in degrees, then convert to meters approximately)
    pixel_size_deg = transform[0]
    pixel_size_m = pixel_size_deg * 111000  # ~111km per degree at this latitude
    
    print(f"\n✓ Loaded DEM")
    print(f"  Shape: {elevation.shape}")
    print(f"  Pixel size: ~{pixel_size_m:.1f} m")

# Mask nodata
nodata = src.nodata
elevation_masked = np.ma.masked_equal(elevation, nodata)

print(f"\nElevation statistics:")
print(f"  Min: {elevation_masked.min():.1f} m")
print(f"  Max: {elevation_masked.max():.1f} m")
print(f"  Mean: {elevation_masked.mean():.1f} m")
print(f"  Std: {elevation_masked.std():.1f} m")

# Calculate gradients (elevation change)
dy, dx = np.gradient(elevation_masked)

# Calculate slope (in degrees)
slope_rad = np.arctan(np.sqrt(dx**2 + dy**2) / pixel_size_m)
slope_deg = np.degrees(slope_rad)

# Handle masked values
slope_deg = np.ma.filled(slope_deg, 0)

print(f"\nSlope statistics:")
print(f"  Min: {slope_deg.min():.2f}°")
print(f"  Max: {slope_deg.max():.2f}°")  
print(f"  Mean: {slope_deg.mean():.2f}°")

# Calculate aspect (direction of slope, in degrees from north)
aspect_rad = np.arctan2(-dy, dx)
aspect_deg = np.degrees(aspect_rad)
aspect_deg = (90 - aspect_deg) % 360  # Convert to compass bearing

aspect_deg = np.ma.filled(aspect_deg, 0)

print(f"\nAspect statistics:")
print(f"  Min: {aspect_deg.min():.1f}°")
print(f"  Max: {aspect_deg.max():.1f}°")
print(f"  Mean: {aspect_deg.mean():.1f}°")

# Calculate Topographic Wetness Index (TWI)
# TWI = ln(a/tan(slope))
# where a = contributing area (simplified as constant for now)
# Higher TWI = more likely to accumulate water

contributing_area = 100  # Simplified assumption
slope_rad_safe = np.maximum(slope_rad, 0.001)  # Avoid division by zero
twi = np.log(contributing_area / np.tan(slope_rad_safe))

print(f"\nTopographic Wetness Index:")
print(f"  Min: {twi.min():.2f}")
print(f"  Max: {twi.max():.2f}")
print(f"  Mean: {twi.mean():.2f}")

# Save slope
profile.update(dtype=rasterio.float32)

with rasterio.open(
    'data/raw/geospatial/leaf_river_slope.tif',
    'w',
    **profile
) as dst:
    dst.write(slope_deg.astype(rasterio.float32), 1)

print(f"\n✓ Saved slope: data/raw/geospatial/leaf_river_slope.tif")

# Save aspect
with rasterio.open(
    'data/raw/geospatial/leaf_river_aspect.tif',
    'w',
    **profile
) as dst:
    dst.write(aspect_deg.astype(rasterio.float32), 1)

print(f"✓ Saved aspect: data/raw/geospatial/leaf_river_aspect.tif")

# Save TWI
with rasterio.open(
    'data/raw/geospatial/leaf_river_twi.tif',
    'w',
    **profile
) as dst:
    dst.write(twi.astype(rasterio.float32), 1)

print(f"✓ Saved TWI: data/raw/geospatial/leaf_river_twi.tif")

# Extract basin-average values (simplified - use basin center)
center_row = elevation.shape[0] // 2
center_col = elevation.shape[1] // 2

basin_elevation = float(elevation[center_row, center_col])
basin_slope = float(slope_deg[center_row, center_col])
basin_twi = float(twi[center_row, center_col])

print(f"\nBasin representative values (center point):")
print(f"  Elevation: {basin_elevation:.1f} m")
print(f"  Slope: {basin_slope:.2f}°")
print(f"  TWI: {basin_twi:.2f}")

# Save terrain features summary
import pandas as pd

terrain_features = pd.DataFrame({
    'feature': ['elevation_m', 'slope_deg', 'twi'],
    'value': [basin_elevation, basin_slope, basin_twi]
})

Path("data/processed").mkdir(parents=True, exist_ok=True)
terrain_features.to_csv("data/processed/terrain_features.csv", index=False)

print(f"\n✓ Saved terrain features: data/processed/terrain_features.csv")

print("\n" + "=" * 60)
print("DEM PROCESSING COMPLETE")
print("=" * 60)