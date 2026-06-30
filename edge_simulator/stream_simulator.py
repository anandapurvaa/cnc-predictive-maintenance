import time
import random
import json
import os
import pandas as pd
from google.cloud import pubsub_v1

# Local Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ai4i2020.csv')
KEY_PATH = os.path.join(os.path.dirname(__file__), '..', 'gcp-key.json')

# GCP Configuration - CHANGE THESE TO MATCH YOUR GCP PROJECT
GCP_PROJECT_ID = "virtual-metrics-501014-f4"  
PUBSUB_TOPIC_ID = "cnc-telemetry-stream"

# Explicitly tell Python where your GCP security credentials live
if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
else:
    print(f"Warning: Local gcp-key.json not found at {KEY_PATH}. Ensure cloud permissions are set.")

def add_sensor_noise(value, variance=0.05):
    return round(value + random.uniform(-variance, variance), 2)

def simulate_and_publish():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Base dataset not found at {DATA_PATH}.")
        return

    # Initialize the GCP Pub/Sub Publisher Client
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TOPIC_ID)

    print(f"Reading baseline dataset... Target GCP Topic: {topic_path}")
    df = pd.read_csv(DATA_PATH)
    
    print("Streaming and publishing to GCP... Press Ctrl+C to stop.")
    
    for _, row in df.iterrows():
        payload = {
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "machine_id": int(row["UDI"]),
            "product_id": row["Product ID"],
            "type": row["Type"],
            "air_temperature_c": add_sensor_noise(row["Air temperature [K]"] - 273.15),
            "process_temperature_c": add_sensor_noise(row["Process temperature [K]"] - 273.15),
            "rotational_speed_rpm": int(row["Rotational speed [rpm]"]),
            "torque_nm": round(row["Torque [Nm]"], 2),
            "tool_wear_min": int(row["Tool wear [min]"]),
            "failure_target": int(row["Machine failure"])
        }
        
        # 1% chance of packet data degradation (simulating network dropping a value)
        if random.random() < 0.01:
            payload["tool_wear_min"] = None
            
        # Convert dictionary to a string, then encode it to raw bytes for transmission
        data_string = json.dumps(payload)
        data_bytes = data_string.encode("utf-8")
        
        try:
            # Publish data packet up to Google Cloud
            future = publisher.publish(topic_path, data_bytes)
            print(f"Published message ID: {future.result()} | Temp: {payload['air_temperature_c']}°C")
        except Exception as e:
            print(f"Failed to publish data to GCP: {e}")
            
        time.sleep(1.5)

if __name__ == "__main__":
    simulate_and_publish()