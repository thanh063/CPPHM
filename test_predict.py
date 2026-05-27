#!/usr/bin/env python3
"""Test prediction with different parameters"""
import requests
import json
import time
import subprocess
import sys

# Start Flask app
print("Khởi động Flask app...")
p = subprocess.Popen([sys.executable, "app.py"])
time.sleep(3)

try:
    # Test 1
    payload1 = {
        "area": 80,
        "district": "q7",
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 4,
        "house_type": "nhà phố"
    }
    print("\n=== Test 1: 80m², 3 phòng, 2 tắm, 4 tầng ===")
    r = requests.post("http://127.0.0.1:5000/predict", json=payload1)
    print(json.dumps(r.json(), indent=2))
    
    # Test 2 - khác diện tích
    payload2 = {
        "area": 150,
        "district": "q7",
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 4,
        "house_type": "nhà phố"
    }
    print("\n=== Test 2: 150m² (diện tích khác), 3 phòng, 2 tắm, 4 tầng ===")
    r = requests.post("http://127.0.0.1:5000/predict", json=payload2)
    print(json.dumps(r.json(), indent=2))
    
    # Test 3 - khác số phòng
    payload3 = {
        "area": 80,
        "district": "q7",
        "bedrooms": 5,
        "bathrooms": 3,
        "floors": 4,
        "house_type": "nhà phố"
    }
    print("\n=== Test 3: 80m², 5 phòng (phòng khác), 3 tắm, 4 tầng ===")
    r = requests.post("http://127.0.0.1:5000/predict", json=payload3)
    print(json.dumps(r.json(), indent=2))
    
    # Test 4 - khác quận
    payload4 = {
        "area": 80,
        "district": "binh_thanh",
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 4,
        "house_type": "nhà phố"
    }
    print("\n=== Test 4: 80m² (Bình Thạnh), 3 phòng, 2 tắm, 4 tầng ===")
    r = requests.post("http://127.0.0.1:5000/predict", json=payload4)
    print(json.dumps(r.json(), indent=2))
    
finally:
    p.terminate()
    p.wait()
    print("\n✅ Đã tắt Flask app")
