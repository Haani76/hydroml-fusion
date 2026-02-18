"""
Download real streamflow data from USGS
Basin: Leaf River near Collins, MS (Site 02472000)
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("DOWNLOADING REAL USGS STREAMFLOW DATA")
print("=" * 60)

# USGS site
site_id = "02472000"
basin_name = "Leaf River"

print(f"\nBasin: {basin_name}")
print(f"USGS Site: {site_id}")

# Build URL for USGS API
url = (
    f"https://waterservices.usgs.gov/nwis/dv/"
    f"?format=json"
    f"&sites={site_id}"
    f"&startDT=2010-01-01"
    f"&endDT=2020-12-31"
    f"&parameterCd=00060"
    f"&siteStatus=all"
)

print(f"\nAttempting download from USGS...")
print(f"URL: {url[:80]}...")

try:
    # Add timeout to detect connection issues quickly
    response = requests.get(url, timeout=30)
    
    if response.status_code == 200:
        print("✓ Download successful!")
        
        # Parse JSON
        data = response.json()
        
        # Extract time series
        ts = data['value']['timeSeries'][0]['values'][0]['value']
        
        # Create DataFrame
        df = pd.DataFrame(ts)
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.rename(columns={'dateTime': 'date', 'value': 'streamflow_cfs'})
        df = df[['date', 'streamflow_cfs']]
        
        # Remove NaN values
        df = df.dropna()
        
        # Save to CSV
        Path("data/raw/streamflow").mkdir(parents=True, exist_ok=True)
        output_file = f"data/raw/streamflow/{site_id}_streamflow.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved to: {output_file}")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Mean flow: {df['streamflow_cfs'].mean():.2f} cfs")
        print(f"  Min flow: {df['streamflow_cfs'].min():.2f} cfs")
        print(f"  Max flow: {df['streamflow_cfs'].max():.2f} cfs")
        
        print("\n" + "=" * 60)
        print("PATH A: SUCCESS - Real USGS data downloaded!")
        print("=" * 60)
        
    else:
        print(f"✗ Error: HTTP {response.status_code}")
        raise Exception("Download failed")
        
except requests.exceptions.Timeout:
    print("\n✗ Connection timeout - network issue detected")
    print("\n" + "=" * 60)
    print("PATH A: FAILED - Will switch to Path B")
    print("=" * 60)
    raise
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\n" + "=" * 60)
    print("PATH A: FAILED - Will switch to Path B")
    print("=" * 60)
    raise