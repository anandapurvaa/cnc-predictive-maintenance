import time
import random
import json
import os
import pandas as pd
import pickle
import numpy as np
from google.cloud import pubsub_v1

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ai4i2020.csv')
KEY_PATH = os.path.join(os.path.dirname(__file__), '..', 'gcp-key.json')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_cnc_model.pkl')

# GCP Configuration
GCP_PROJECT_ID = "virtual-metrics-501014-f4"  
PUBSUB_TOPIC_ID = "cnc-telemetry-stream"

if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

def load_ml_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model artifact not found at {MODEL_PATH}. Run training script first.")
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def smart_stream():
    model = load_ml_model()
    if model is None or not os.path.exists(DATA_PATH):
        return

    # Initialize Pub/Sub Publisher
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TOPIC_ID)
    
    df = pd.read_csv(DATA_PATH)
    print("🚀 Starting Smart Edge Stream with Live ML Predictions... Press Ctrl+C to stop.")

    for _, row in df.iterrows():
        # Clean/Simulate data matching training features
        air_temp = round(row["Air temperature [K]"] - 273.15 + random.uniform(-0.05, 0.05), 2)
        proc_temp = round(row["Process temperature [K]"] - 273.15 + random.uniform(-0.05, 0.05), 2)
        temp_diff = round(proc_temp - air_temp, 2)
        rot_speed = int(row["Rotational speed [rpm]"])
        torque = round(row["Torque [Nm]"], 2)
        tool_wear = int(row["Tool wear [min]"])
        
        # Feature Engineering Flags for Model Input
        is_high_speed = 1 if rot_speed > 2500 else 0
        is_critical_wear = 1 if tool_wear >= 200 else 0

        # Construct feature array for model prediction (Order must match training features!)
        features = np.array([[air_temp, proc_temp, temp_diff, rot_speed, torque, tool_wear, is_high_speed, is_critical_wear]])
        
        # Run Edge Inference
        fail_probability = float(model.predict_proba(features)[0][1])
        ai_prediction_flag = 1 if fail_probability > 0.5 else 0

        # Build payload with live analytics attached
        payload = {
            "timestamp_utc": pd.Timestamp.now('UTC').isoformat(),
            "machine_id": int(row["UDI"]),
            "product_id": row["Product ID"],
            "type": row["Type"],
            "air_temperature_c": air_temp,
            "process_temperature_c": proc_temp,
            "rotational_speed_rpm": rot_speed,
            "torque_nm": torque,
            "tool_wear_min": tool_wear,
            "failure_target": int(row["Machine failure"]),
            # New MLOps fields!
            "ml_failure_probability": round(fail_probability, 4),
            "ml_prediction_lead": ai_prediction_flag
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        
        try:
            future = publisher.publish(topic_path, data_bytes)
            alert_status = "⚠️ CRITICAL FAILURE RISK" if ai_prediction_flag == 1 else "✅ Normal Operations"
            print(f"Sent ID: {future.result()} | ML Risk Calc: {round(fail_probability*100, 2)}% -> {alert_status}")
        except Exception as e:
            print(f"Failed to stream: {e}")

        time.sleep(1)

if __name__ == "__main__":
    smart_stream()