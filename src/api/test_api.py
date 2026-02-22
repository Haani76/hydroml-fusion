"""
Test the FastAPI endpoints
"""

import requests
import json

print("=" * 60)
print("TESTING HYDROML-FUSION API")
print("=" * 60)

base_url = "http://localhost:8000"

# Test 1: Health check
print("\n1. Testing health endpoint...")
try:
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")
    print("   Make sure API is running: uvicorn main:app")

# Test 2: Root endpoint
print("\n2. Testing root endpoint...")
try:
    response = requests.get(f"{base_url}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Info endpoint
print("\n3. Testing info endpoint...")
try:
    response = requests.get(f"{base_url}/info")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Project: {data.get('project')}")
    print(f"   Basin: {data.get('basin')}")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Prediction endpoint
print("\n4. Testing prediction endpoint...")
try:
    payload = {
        "precipitation_mm": 10.5,
        "temperature_c": 22.0,
        "pet_mm": 3.5
    }
    
    response = requests.post(
        f"{base_url}/predict",
        json=payload
    )
    
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Predicted streamflow: {result['streamflow_mm']:.2f} mm/day")
        print(f"   Predicted streamflow: {result['streamflow_cfs']:.2f} cfs")
        print(f"   Model: {result['model']}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("API TESTING COMPLETE")
print("=" * 60)