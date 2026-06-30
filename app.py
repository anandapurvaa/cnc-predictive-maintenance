import streamlit as st
import os
import json
import random
import time
import pandas as pd
import numpy as np
import pickle
import subprocess
from google.cloud import bigquery
import plotly.express as px

# Setup GCP configuration
KEY_PATH = os.path.join(os.path.dirname(__file__), 'gcp-key.json')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_cnc_model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ai4i2020.csv')

if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

# Page Config
st.set_page_config(page_title="CNC Predictive Maintenance Control Room", layout="wide")
st.title("🏭 CNC Predictive Maintenance Control Room")
st.markdown("An End-to-End MLOps & Data Engineering Production Portfolio")

# Load ML Model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# Create Tabs
tab1, tab2, tab3 = st.tabs(["🎮 Recruiter Interactive Demo", "📊 Live Data Warehouse Analytics", "🎛️ What-If Edge Model Simulator"])

# TAB 1: RECRUITER LIVE PIPELINE SIMULATOR
with tab1:
    st.header("🚀 Live Pipeline Interactive Simulator")
    st.write("""
        **Hey Recruiter!** You don't need to run any code locally. Click the button below to simulate live CNC IoT sensors. 
        This will generate fresh data, run edge ML inference, stream it to Google BigQuery, and trigger a dbt transformation run in real-time!
    """)
    
    stream_count = st.slider("Number of telemetry rows to stream into cloud pipeline", 5, 30, 10)
    
    if st.button("🔥 Run End-to-End Cloud Pipeline"):
        if model is None or not os.path.exists(DATA_PATH):
            st.error("Missing model artifact or raw data CSV file.")
        else:
            # 1. Simulate Edge Streaming Directly to BigQuery Raw Table
            st.markdown("### 📡 Step 1: Simulating Live Telemetry & Edge Inference...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                bq_client = bigquery.Client()
                # Target the raw table directly to bypass Pub/Sub delays for instant portfolio execution
                table_ref = bq_client.dataset("raw_factory_data").table("cnc_telemetry_raw")
                
                df_source = pd.read_csv(DATA_PATH).sample(stream_count)
                rows_to_insert = []
                
                for idx, (_, row) in enumerate(df_source.iterrows()):
                    # Match exact feature engineering pipeline
                    air_temp = round(row["Air temperature [K]"] - 273.15 + random.uniform(-0.05, 0.05), 2)
                    proc_temp = round(row["Process temperature [K]"] - 273.15 + random.uniform(-0.05, 0.05), 2)
                    temp_diff = round(proc_temp - air_temp, 2)
                    rot_speed = int(row["Rotational speed [rpm]"])
                    torque = round(row["Torque [Nm]"], 2)
                    tool_wear = int(row["Tool wear [min]"])
                    
                    is_high_speed = 1 if rot_speed > 2500 else 0
                    is_critical_wear = 1 if tool_wear >= 200 else 0
                    
                    features = np.array([[air_temp, proc_temp, temp_diff, rot_speed, torque, tool_wear, is_high_speed, is_critical_wear]])
                    fail_probability = float(model.predict_proba(features)[0][1])
                    
                    # Package payload matching warehouse schema fields
                    payload = {
                        "timestamp_utc": pd.Timestamp.now('UTC').isoformat(),
                        "machine_id": int(row["UDI"]),
                        "product_id": str(row["Product ID"]),
                        "type": str(row["Type"]),
                        "air_temperature_c": float(air_temp),
                        "process_temperature_c": float(proc_temp),
                        "rotational_speed_rpm": int(rot_speed),
                        "torque_nm": float(torque),
                        "tool_wear_min": int(tool_wear),
                        "failure_target": int(row["Machine failure"]),
                        "ml_failure_probability": float(fail_probability),
                        "ml_prediction_lead": int(1 if fail_probability > 0.5 else 0)
                    }
                    rows_to_insert.append(payload)
                    
                    # Update progress UI
                    time.sleep(0.1)
                    progress_bar.progress((idx + 1) / stream_count)
                    status_text.text(f"Processed row {idx+1}/{stream_count} | ML Failure Risk calculated: {round(fail_probability*100,1)}%")
                
                errors = bq_client.insert_rows_json(table_ref, rows_to_insert)
                if errors == []:
                    st.success(f"✅ Successfully streamed {stream_count} rows to Google BigQuery!")
                else:
                    st.error(f"BQ Streaming Error: {errors}")
                
                # 2. Trigger dbt Transformation Models Programmatically
                st.markdown("### ⚙️ Step 2: Running dbt Cloud Transformation Compilation Layer...")
                with st.spinner("Executing 'dbt run' to build production warehouse analytics marts..."):
                    result = subprocess.run(
                        [".venv/Scripts/dbt", "run", "--project-dir", "cnc_transformation"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        st.success("✅ dbt transformation compilation completed successfully!")
                        st.balloons()
                        st.info("💡 Switch to the **Live Data Warehouse Analytics** tab above to see your new data fully mapped!")
                    else:
                        st.error("dbt compilation execution failed.")
                        st.code(result.stderr)
                        
            except Exception as e:
                st.error(f"Pipeline Execution Failed: {e}")

# TAB 2: BigQuery Analytics
# TAB 2: BigQuery Analytics
with tab2:
    st.header("Cloud Warehouse Telemetry")
    try:
        client = bigquery.Client()
        query = """
            SELECT reading_at, machine_id, air_temperature_c, process_temperature_c, 
                   temperature_differential_c, rotational_speed_rpm, torque_nm, 
                   tool_wear_min, ai_failure_risk_score, has_failed
            FROM `virtual-metrics-501014-f4.raw_factory_data.fct_cnc_failures`
            ORDER BY reading_at DESC LIMIT 100
        """
        df = client.query(query).to_dataframe()
    except Exception as e:
        df = pd.DataFrame() # Graceful fallback to empty dataframe if credentials/connection fail

    # 🌟 PORTFOLIO AUTOMATIC FALLBACK LAYER
    # If BigQuery is empty, load local CSV data as a baseline preview so visuals render instantly
    if df.empty and os.path.exists(DATA_PATH):
        st.caption("⚠️ Note: Displaying historical baseline sample (Cloud warehouse table is empty...)")
        df_raw = pd.read_csv(DATA_PATH).sample(100, random_state=42)
        
        # Structure the baseline DataFrame to match your warehouse schema expectations
        df = pd.DataFrame()
        
        # Clean string-formatted ISO timestamps that Plotly parses perfectly
        timestamps = pd.date_range(end=pd.Timestamp.now('UTC'), periods=100, freq='min')
        df['reading_at'] = timestamps.strftime('%Y-%m-%d %H:%M:%S') 
        
        # Cast Machine ID to a string so the Plotly chart treats them as distinct categorical entities
        df['machine_id'] = df_raw['UDI'].astype(str)
        df['air_temperature_c'] = round(df_raw["Air temperature [K]"] - 273.15, 2)
        df['process_temperature_c'] = round(df_raw["Process temperature [K]"] - 273.15, 2)
        df['temperature_differential_c'] = round(df['process_temperature_c'] - df['air_temperature_c'], 2)
        df['rotational_speed_rpm'] = df_raw["Rotational speed [rpm]"]
        df['torque_nm'] = df_raw["Torque [Nm]"]
        df['tool_wear_min'] = df_raw["Tool wear [min]"]
        df['ai_failure_risk_score'] = 0.06 # Default flat baseline guess
        df['has_failed'] = df_raw["Machine failure"]

    # Render dashboard components only if data is successfully loaded (via BigQuery or local CSV)
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Warehouse Rows Logged", len(df))
        with col2:
            st.metric("Avg Fleet Tool Wear", f"{round(df['tool_wear_min'].mean(), 1)} min")
        with col3:
            st.metric("Total Pipeline Failures Caught", int(df['has_failed'].sum()))
        
        st.subheader("Sensor Temperature Differentials Over Time")
        df_sorted = df.sort_values('reading_at')
        fig = px.line(
            df_sorted, x='reading_at', y='temperature_differential_c', color='machine_id',
            labels={'reading_at': 'Timestamp (UTC)', 'temperature_differential_c': 'Thermal Delta (°C)'},
            title="Active Thermal Degradation (Process Temp - Air Temp)", template="plotly_white"
        )
        fig.update_layout(hovermode="x unified", legend_title_text="Machine Asset ID")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Raw Analytics Fleet Log Entries (fct_cnc_failures)")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data available. Go to Tab 1 to execute the live stream simulator pipeline!")

# TAB 3: Model Simulator
with tab3:
    st.header("Interactive Edge ML Inference")
    if model is not None:
        st.write("Manipulate factory equipment sensors below to calculate live failure risk probabilities immediately.")
        col1, col2 = st.columns(2)
        with col1:
            air_temp = st.slider("Air Temperature (°C)", 20.0, 45.0, 30.0)
            proc_temp = st.slider("Process Temperature (°C)", 25.0, 60.0, 40.0)
            rot_speed = st.slider("Rotational Speed (RPM)", 1000, 3000, 1500)
            torque = st.slider("Torque (Nm)", 10.0, 80.0, 40.0)
            tool_wear = st.slider("Tool Wear Accumulation (min)", 0, 250, 60)
        
        with col2:
            # Calculate metrics
            temp_diff = round(proc_temp - air_temp, 2)
            is_high_speed = 1.0 if rot_speed > 2500 else 0.0
            is_critical_wear = 1.0 if tool_wear >= 200 else 0.0
            
            # Build the DataFrame with explicit float casting
            feature_dict = {
                "air_temperature_c": [float(air_temp)],
                "process_temperature_c": [float(proc_temp)],
                "temperature_differential_c": [float(temp_diff)],
                "rotational_speed_rpm": [float(rot_speed)],
                "torque_nm": [float(torque)],
                "tool_wear_min": [float(tool_wear)],
                "is_high_speed_anomalous": [float(is_high_speed)],
                "is_critical_wear_risk": [float(is_critical_wear)]
            }
            features_df = pd.DataFrame(feature_dict).astype("float64")
            
            # Enforce the strict order
            expected_order = [
                "air_temperature_c", "process_temperature_c", "temperature_differential_c",
                "rotational_speed_rpm", "torque_nm", "tool_wear_min",
                "is_high_speed_anomalous", "is_critical_wear_risk"
            ]
            features_df = features_df[expected_order]
            
            # Inference Execution
            prob = model.predict_proba(features_df)[0][1]
            risk_pct = round(prob * 100, 2)
            
            st.subheader("Live Diagnostic Result")
            if risk_pct < 30:
                st.success(f"🟢 Asset Status: HEALTHY ({risk_pct}%)")
            elif risk_pct < 70:
                st.warning(f"🟡 Asset Status: DEGRADED PERFORMANCE ({risk_pct}%)")
            else:
                st.error(f"🔴 Asset Status: CRITICAL RISK FAILURE DETECTED ({risk_pct}%)")
            st.metric(label="Calculated Risk Weight", value=f"{risk_pct}%")