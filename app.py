import os
import json
import numpy as np
import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="E-Commerce Fraud Engine", layout="wide")
st.title("🛡️ Real-Time E-Commerce Fraud & Risk Engine")

# 1. Secure Authentication Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "config", "gcp_credentials.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

# 2. Cache resources for performance
@st.cache_resource
def init_gcp_clients():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "config", "gcp_credentials.json")
    
    # 🔐 Hybrid Authentication Check
    if os.path.exists(CREDENTIALS_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
        bq_client = bigquery.Client()
    elif "gcp_service_account" in st.secrets:
        secrets_dict = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(secrets_dict)
        bq_client = bigquery.Client(credentials=credentials, project=secrets_dict["project_id"])
    else:
        raise FileNotFoundError("❌ No credential resources located locally or in cloud secrets configuration.")

    # Load machine learning engine weights
    model = XGBClassifier()
    model.load_model(os.path.join(PROJECT_ROOT, "artifacts", "fraud_risk_xgb.json"))
    
    return bq_client, model

try:
    bq_client, model = init_gcp_clients()
    PROJECT_ID = bq_client.project
    FULL_TABLE_REF = f"{PROJECT_ID}.fraud_stream.transactions"

    # Sidebar Controls
    st.sidebar.header("🎛️ Risk Threshold Settings")
    threshold = st.sidebar.slider("Fraud Probability Cutoff", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
    
    # Allow looking at a wider window for stable statistical performance tracking
    window_size = st.sidebar.selectbox("Performance Evaluation Window", [20, 50, 100, 500], index=1)
    
    if st.sidebar.button("🔄 Refresh Live Stream"):
        st.rerun()

    # 3. Pull the live stream data + including ground-truth label
    query = f"""
        SELECT timestamp, transaction_id, user_id, amount, velocity_10m, ip_country, billing_country, is_fraud_label 
        FROM `{FULL_TABLE_REF}` 
        ORDER BY timestamp DESC 
        LIMIT {window_size}
    """
    df_live = bq_client.query(query).to_dataframe()

    if not df_live.empty:
        # 4. Feature Engineer Live Data on the Fly
        df_live['country_mismatch'] = (df_live['ip_country'] != df_live['billing_country']).astype(int)
        
        # Isolate target training features
        X_live = df_live[['amount', 'velocity_10m', 'country_mismatch']]
        
        # 5. Live Prediction
        y_prob = model.predict_proba(X_live)[:, 1]
        df_live['Fraud_Probability'] = y_prob
        
        # Binary prediction vector based on the sidebar slider threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        df_live['Decision'] = np.where(y_pred == 1, "🚨 BLOCKED (High Risk)", "🟢 APPROVED")

        # 6. CALCULATE LIVE MODEL PERFORMANCE METRICS
        y_true = df_live['is_fraud_label'].astype(int)
        
        # Use zero_division parameter to prevent dashboard crashes before any fraud events appear
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 7. Layout Dashboard Views
        st.subheader("🎯 Live XGBoost Model Performance Evaluation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Accuracy", f"{acc:.1%}", help="Percentage of overall correct decisions.")
        m2.metric("Live Precision", f"{prec:.1%}", help="Out of all transactions flagged as fraud, how many were actually fraud? (Low values = High false alarm rate)")
        m3.metric("Live Recall", f"{rec:.1%}", help="Out of all actual fraud transactions injected, how many did the model catch? (Low values = Missed attacks)")
        m4.metric("Live F1-Score", f"{f1:.2f}", help="Harmonic mean balancing precision and recall.")

        st.markdown("---")

        # 8. High-Impact Operational Metrics Display (Showing only the top 20 for UI visibility)
        df_display = df_live.head(20).copy()
        total_checked = len(df_live)
        blocked_count = int(np.sum(df_live['Decision'].str.contains("BLOCKED")))
        total_risk_vol = df_live[df_live['Decision'].str.contains("BLOCKED")]['amount'].sum()

        st.subheader("📊 Operational Metrics (Top 20 Transactions Displayed)")
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Active Evaluation Window", f"{total_checked} Txns")
        col2.metric("🛑 Automated Interceptions", f"{blocked_count} Alerts")
        col3.metric("💰 Fraud Volume Intercepted", f"${total_risk_vol:,.2f}")

        # Formatting helper to highlight risk rows vs true labels
        def highlight_fraud(row):
            styles = [''] * len(row)
            is_blocked = "BLOCKED" in str(row['Decision'])
            is_actual_fraud = int(row['is_fraud_label']) == 1
            
            for i, col in enumerate(row.index):
                if is_blocked and is_actual_fraud:
                    styles[i] = 'background-color: #d4edda; color: #155724;' # True Positive (Green highlight for catching it!)
                elif is_blocked and not is_actual_fraud:
                    styles[i] = 'background-color: #fff3cd; color: #856404;' # False Positive (Yellow highlight for false alarm)
                elif not is_blocked and is_actual_fraud:
                    styles[i] = 'background-color: #f8d7da; color: #721c24;' # False Negative (Red highlight for missed fraud!)
            return styles

        # Reordering columns slightly for pristine readability
        columns_order = [
            'timestamp', 'transaction_id', 'user_id', 'amount', 
            'velocity_10m', 'country_mismatch', 'is_fraud_label', 
            'Fraud_Probability', 'Decision'
        ]
        
        styled_df = df_display[columns_order].style.apply(highlight_fraud, axis=1).format({
            "amount": "${:.2f}", 
            "Fraud_Probability": "{:.2%}"
        })
        st.dataframe(styled_df, use_container_width=True)

    else:
        st.warning("No transactions found in the database stream. Start your Cloud Run service to feed data!")

except Exception as e:
    st.error(f"Initialization Error: {e}")