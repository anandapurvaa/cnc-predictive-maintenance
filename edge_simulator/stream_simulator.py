import time
import random
import json
import os
import pandas as pd

# Path configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ai4i2020.csv')

def add_sensor_noise(value, variance=0.05):
    """Simulates real-world sensor jitter."""
    return round(value + random.uniform(-variance, variance), 2)

def simulate_stream():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Base dataset not found at {DATA_PATH}. Please place the CSV file there.")
        return

    print("Reading baseline manufacturing dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print("Starting industrial edge streaming simulation... Press Ctrl+C to stop.")
    
    # Loop through rows to mimic real-time sensor ticks
    for _, row in df.iterrows():
        # Map raw column names to clean, enterprise-ready names
        payload = {
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "machine_id": row["UDI"],
            "product_id": row["Product ID"],
            "type": row["Type"],
            "air_temperature_c": add_sensor_noise(row["Air temperature [K]"] - 273.15), # Convert Kelvin to Celsius
            "process_temperature_c": add_sensor_noise(row["Process temperature [K]"] - 273.15),
            "rotational_speed_rpm": int(row["Rotational speed [rpm]"]),
            "torque_nm": round(row["Torque [Nm]"], 2),
            "tool_wear_min": int(row["Tool wear [min]"]),
            "failure_target": int(row["Machine failure"]) # Real labels for validation later
        }
        
        # Human touch: Simulate sporadic network packet drops (1% of the time, drop tool wear reading)
        if random.random() < 0.01:
            payload["tool_wear_min"] = None
            
        print(json.dumps(payload))
        time.sleep(1.5) # Wait 1.5 seconds between ticks

if __name__ == "__main__":
    simulate_stream()