"""
Download real SRTM DEM from OpenTopography with API key
"""

import requests
import rasterio
import numpy as np
from pathlib import Path

print("=" * 60)
print("DOWNLOADING REAL SRTM DEM")
print("=" * 60)

# Leaf River basin bounds
min_lon, max_lon = -90.0, -88.5
min_lat, max_lat = 31.0, 32.5

print(f"\nBasin bounds:")
print(f"  Longitude: {min_lon} to {max_lon}")
print(f"  Latitude: {min_lat} to {max_lat}")

# OpenTopography API with your key
url = "https://portal.opentopography.org/API/globaldem"

params = {
    'demtype': 'SRTMGL3',  # SRTM 90m
    'south': min_lat,
    'north': max_lat,
    'west': min_lon,
    'east': max_lon,
    'outputFormat': 'GTiff',
    'API_Key': '751816c60081e23d37e5a35f81922e26'
}

print(f"\nDownloading from OpenTopography...")
print(f"DEM type: SRTM 90m (SRTMGL3)")

try:
    response = requests.get(url, params=params, timeout=120)
    
    if response.status_code == 200:
        print("✓ Download successful!")
        
        # Save DEM
        Path("data/raw/geospatial").mkdir(parents=True, exist_ok=True)
        output_file = "data/raw/geospatial/leaf_river_dem.tif"
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        # Read and display info
        with rasterio.open(output_file) as src:
            elevation = src.read(1)
            
            # Mask nodata values
            elevation_masked = np.ma.masked_equal(elevation, src.nodata)
            
            print(f"\n✓ Saved DEM to: {output_file}")
            print(f"\nDEM Statistics:")
            print(f"  Shape: {elevation.shape} pixels")
            print(f"  Min elevation: {elevation_masked.min():.1f} m")
            print(f"  Max elevation: {elevation_masked.max():.1f} m")
            print(f"  Mean elevation: {elevation_masked.mean():.1f} m")
            print(f"  CRS: {src.crs}")
            print(f"  Bounds: {src.bounds}")
        
        print("\n" + "=" * 60)
        print("REAL SRTM DEM DOWNLOADED!")
        print("=" * 60)
        
    else:
        print(f"✗ Error: HTTP {response.status_code}")
        print(f"Response: {response.text[:500]}")
        raise Exception("Download failed")
        
except Exception as e:
    print(f"\n✗ Error: {e}")
    raise