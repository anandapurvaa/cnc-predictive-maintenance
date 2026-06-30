import streamlit as st
import os
import random
import time
import pandas as pd
import numpy as np
import pickle
from google.cloud import bigquery
import plotly.express as px
from dbt.cli.main import dbtRunner

# Explicitly map Streamlit Secrets to System Environment Variables
if "gcp" in st.secrets:
    os.environ["STREAMLIT_GCP_PROJECT_ID"] = str(st.secrets["gcp"]["project_id"])

# Now, dbt will see the variable when it initializes
# 1. Setup Base File Directories & Fallbacks
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_PATH = os.path.join(BASE_DIR, 'gcp-key.json')

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_cnc_model.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'models/xgboost_cnc_model.pkl'

DATA_PATH = os.path.join(BASE_DIR, 'data', 'ai4i2020.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/ai4i2020.csv'

# 2. Authenticate and Initialize BigQuery Client
bq_client = None

if "gcp" in st.secrets:
    try:
        gcp_info = dict(st.secrets["gcp"])
        bq_client = bigquery.Client.from_service_account_info(gcp_info)
    except Exception as e:
        st.error(f"Streamlit Cloud Secrets authentication error: {e}")
elif os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
    try:
        bq_client = bigquery.Client()
    except Exception as e:
        st.error(f"Local Service Account authentication error: {e}")

# 3. Load Machine Learning Model 
@st.cache_resource
def load_ml_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_ml_model()

# 4. Configure Application View Layout
st.set_page_config(page_title="CNC Predictive Maintenance Control Room", layout="wide")
st.title("🏭 CNC Predictive Maintenance Control Room")
st.markdown("An End-to-End MLOps & Data Engineering Production Portfolio")

tab1, tab2, tab3 = st.tabs(["🎮 Recruiter Interactive Demo", "📊 Live Data Warehouse Analytics", "🎛️ What-If Edge Model Simulator"])

# TAB 1: DATA INGESTION PIPELINE PIPELINE SIMULATOR
with tab1:
    st.header("🚀 Live Pipeline Interactive Simulator")
    st.write("""
        **Hey Recruiter!** You don't need to run any code locally. Click the button below to simulate live CNC IoT sensors. 
        This will generate fresh data, run edge ML inference, and stream it straight to Google BigQuery in real-time!
    """)
    
    stream_count = st.slider("Number of telemetry rows to stream into cloud pipeline", 5, 30, 10)
    
    if st.button("🔥 Run End-to-End Cloud Pipeline"):
        if model is None or not os.path.exists(DATA_PATH):
            st.error("Missing required model artifact or baseline raw dataset file.")
        elif bq_client is None:
            st.error("BigQuery client is unauthenticated. Check your Streamlit Secrets.")
        else:
            st.markdown("### 📡 Step 1: Simulating Live Telemetry & Edge Inference...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Target the landing table directly
                table_ref = bq_client.dataset("raw_factory_data").table("cnc_telemetry_raw")
                
                df_source = pd.read_csv(DATA_PATH).sample(stream_count)
                rows_to_insert = []
                session_tracking_rows = []
                
                for idx, (_, row) in enumerate(df_source.iterrows()):
                    air_temp = round(row["Air temperature [K]"] - 273.15 + random.uniform(-0.05, 0.05), 2)
                    proc_temp = round(row["Process temperature [K]"] - 273.15 + random.uniform(-0.05, 0.05), 2)
                    temp_diff = round(proc_temp - air_temp, 2)
                    rot_speed = int(row["Rotational speed [rpm]"])
                    torque = round(row["Torque [Nm]"], 2)
                    tool_wear = int(row["Tool wear [min]"])
                    
                    is_high_speed = 1 if rot_speed > 2500 else 0
                    is_critical_wear = 1 if tool_wear >= 200 else 0
                    
                    # Machine Learning Feature Vector Structure Alignment
                    features = np.array([[air_temp, proc_temp, temp_diff, rot_speed, torque, tool_wear, is_high_speed, is_critical_wear]])
                    fail_probability = float(model.predict_proba(features)[0][1])
                    
                    current_time_str = pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M:%S')
                    
                    payload = {
                        "timestamp_utc": current_time_str,
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
                    
                    # Store data structured exactly like our warehouse target output view format
                    session_tracking_rows.append({
                        "reading_at": current_time_str,
                        "machine_id": str(row["UDI"]),
                        "air_temperature_c": float(air_temp),
                        "process_temperature_c": float(proc_temp),
                        "temperature_differential_c": float(temp_diff),
                        "rotational_speed_rpm": int(rot_speed),
                        "torque_nm": float(torque),
                        "tool_wear_min": int(tool_wear),
                        "ai_failure_risk_score": float(fail_probability),
                        "has_failed": int(row["Machine failure"])
                    })
                    
                    time.sleep(0.05)
                    progress_bar.progress((idx + 1) / stream_count)
                    status_text.text(f"Processed row {idx+1}/{stream_count} | ML Failure Risk: {round(fail_probability*100,1)}%")
                
                # Streaming Ingestion API call directly to Google BigQuery
                errors = bq_client.insert_rows_json(table_ref, rows_to_insert)
                if errors == []:
                    st.success(f"✅ Successfully streamed {stream_count} rows to Google BigQuery!")
                    st.balloons()
                    
                    # Update local state memory instantly so charts update without a single page lag!
                    new_live_df = pd.DataFrame(session_tracking_rows)
                    if "live_stream_cache" not in st.session_state:
                        st.session_state["live_stream_cache"] = new_live_df
                    else:
                        st.session_state["live_stream_cache"] = pd.concat([st.session_state["live_stream_cache"], new_live_df], ignore_index=True)
                    
                    st.info("💡 Switch to the **Live Data Warehouse Analytics** tab above to view your entries!")
                else:
                    st.error(f"BigQuery Ingestion Exception Log: {errors}")
                        
            except Exception as e:
                st.error(f"Pipeline Ingestion Fault: {e}")

# TAB 2: LIVE METRICS WAREHOUSE DASHBOARD
with tab2:
    st.header("Cloud Warehouse Telemetry")
    
    df = pd.DataFrame()
    if bq_client is not None:
        try:
            # Query directly out of our logs table to pull live updates dynamically
            query = """
                SELECT timestamp_utc as reading_at, machine_id, air_temperature_c, process_temperature_c, 
                       (process_temperature_c - air_temperature_c) as temperature_differential_c, rotational_speed_rpm, torque_nm, 
                       tool_wear_min, ml_failure_probability as ai_failure_risk_score, failure_target as has_failed
                FROM `virtual-metrics-501014-f4.raw_factory_data.cnc_telemetry_raw`
                ORDER BY timestamp_utc DESC LIMIT 150
            """
            df = bq_client.query(query).to_dataframe()
            if not df.empty:
                df['machine_id'] = df['machine_id'].astype(str)
        except Exception as e:
            df = pd.DataFrame()

    # Fallback to local session aggregation data to bridge visual logs seamlessly
    if df.empty and os.path.exists(DATA_PATH):
        df_raw = pd.read_csv(DATA_PATH).sample(100, random_state=42)
        df_hist = pd.DataFrame()
        timestamps = pd.date_range(end=pd.Timestamp.now('UTC') - pd.Timedelta(hours=1), periods=100, freq='min')
        
        df_hist['reading_at'] = timestamps.strftime('%Y-%m-%d %H:%M:%S') 
        df_hist['machine_id'] = df_raw['UDI'].astype(str)
        df_hist['air_temperature_c'] = round(df_raw["Air temperature [K]"] - 273.15, 2)
        df_hist['process_temperature_c'] = round(df_raw["Process temperature [K]"] - 273.15, 2)
        df_hist['temperature_differential_c'] = round(df_hist['process_temperature_c'] - df_hist['air_temperature_c'], 2)
        df_hist['rotational_speed_rpm'] = df_raw["Rotational speed [rpm]"]
        df_hist['torque_nm'] = df_raw["Torque [Nm]"]
        df_hist['tool_wear_min'] = df_raw["Tool wear [min]"]
        df_hist['ai_failure_risk_score'] = 0.05
        df_hist['has_failed'] = df_raw["Machine failure"]

        if "live_stream_cache" in st.session_state:
            df = pd.concat([df_hist, st.session_state["live_stream_cache"]], ignore_index=True)
        else:
            df = df_hist

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
            df_sorted, x='reading_at', y='temperature_differential_c',
            labels={'reading_at': 'Timestamp (UTC)', 'temperature_differential_c': 'Thermal Delta (°C)'},
            title="Active Thermal Degradation (Process Temp - Air Temp)", template="plotly_white"
        )
        fig.update_traces(mode='lines+markers', marker=dict(size=4, color='#FF4B4B'), line=dict(color='#262730'))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Raw Analytics Fleet Log Entries")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data available. Go to Tab 1 to execute the live stream simulator pipeline!")

# TAB 3: EDGE MODEL SIMULATOR INTERFACE
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
            temp_diff = round(proc_temp - air_temp, 2)
            is_high_speed = 1.0 if rot_speed > 2500 else 0.0
            is_critical_wear = 1.0 if tool_wear >= 200 else 0.0
            
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
            
            expected_order = [
                "air_temperature_c", "process_temperature_c", "temperature_differential_c",
                "rotational_speed_rpm", "torque_nm", "tool_wear_min",
                "is_high_speed_anomalous", "is_critical_wear_risk"
            ]
            features_df = features_df[expected_order]
            
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