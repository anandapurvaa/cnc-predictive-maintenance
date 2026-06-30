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

# 1. BRIDGE SECRETS TO ENVIRONMENT (DO NOT REMOVE)
if "gcp" in st.secrets:
    os.environ["STREAMLIT_GCP_PROJECT_ID"] = str(st.secrets["gcp"]["project_id"])

# 2. FILE PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_cnc_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ai4i2020.csv')
EXPECTED_COLUMNS = [
    "air_temperature_c", "process_temperature_c", "temperature_differential_c",
    "rotational_speed_rpm", "torque_nm", "tool_wear_min",
    "is_high_speed_anomalous", "is_critical_wear_risk"
]

# 3. AUTH & MODEL LOADING
bq_client = None
if "gcp" in st.secrets:
    bq_client = bigquery.Client.from_service_account_info(dict(st.secrets["gcp"]))

@st.cache_resource
def load_ml_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f: return pickle.load(f)
    return None

model = load_ml_model()

# 4. APP INTERFACE
st.set_page_config(page_title="CNC Predictive Maintenance Control Room", layout="wide")
st.title("🏭 CNC Predictive Maintenance Control Room")

tab1, tab2, tab3 = st.tabs(["🎮 Pipeline Demo", "📊 Analytics", "🎛️ Edge Simulator"])

with tab1:
    if st.button("🔥 Run End-to-End Pipeline"):
        with st.spinner("Compiling and running dbt transformations..."):
            os.environ["DBT_SEND_ANONYMOUS_USAGE_STATS"] = "False"
            dbt = dbtRunner()
            res = dbt.invoke(["run", "--project-dir", "cnc_transformation", "--profiles-dir", "cnc_transformation"])
            if res.success: st.success("✅ dbt pipeline executed successfully!")
            else: st.error(f"Pipeline failed: {res.exception}")

with tab2:
    if bq_client:
        query = "SELECT * FROM `virtual-metrics-501014-f4.raw_factory_data.cnc_telemetry_raw` ORDER BY timestamp_utc DESC LIMIT 100"
        st.dataframe(bq_client.query(query).to_dataframe())

with tab3:
    # Feature inputs... (Your previous slider logic goes here)
    # Ensure you use EXPECTED_COLUMNS when calling model.predict_proba
    st.write("Simulator ready.")