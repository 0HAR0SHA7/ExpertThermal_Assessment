import requests
from pinn_thermal import MATERIALS

# Flask API URL
API_URL = "http://127.0.0.1:5000/predict"

# Sample inputs for testing
test_cases = [
    {"power": 50, "ambient_temp": 25},
    {"power": 75, "ambient_temp": 30},
    {"power": 100, "ambient_temp": 35}
]

print("Testing PINN API predictions for all materials...\n")

for case in test_cases:
    for material in MATERIALS:
        payload = {
            "power": case["power"],
            "ambient_temp": case["ambient_temp"],
            "material": material
        }
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"Input: {payload}")
                print(f"API Response: {data}")
                print("-" * 50)
            else:
                print(f"Error for input {payload}: {response.text}")
        except Exception as e:
            print(f"Exception for input {payload}: {e}")
