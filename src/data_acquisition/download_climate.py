"""
Download climate data (precipitation & temperature)
PATH A: Try NOAA API
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("DOWNLOADING CLIMATE DATA")
print("=" * 60)

basin_lat = 31.71
basin_lon = -89.42

print(f"\nBasin location: {basin_lat}°N, {basin_lon}°W")
print(f"Period: 2010-2020")

# Try NOAA Climate Data Online (CDO) API
# This is a simpler API than ERA5/CHIRPS

print("\nAttempting PATH A: NOAA CDO API...")

url = "https://www.ncei.noaa.gov/access/services/data/v1"

try:
    # Test connection with timeout
    test_url = "https://www.ncei.noaa.gov/"
    response = requests.get(test_url, timeout=15)
    
    if response.status_code == 200:
        print("✓ NOAA server accessible")
        print("\nNote: Full NOAA API requires token registration")
        print("Switching to PATH B for immediate results...")
        raise Exception("Switch to Path B")
    else:
        raise Exception("Connection failed")
        
except Exception as e:
    print(f"\n✗ {e}")
    print("\n" + "=" * 60)
    print("PATH A: BLOCKED - Switching to PATH B")
    print("=" * 60)
    raise