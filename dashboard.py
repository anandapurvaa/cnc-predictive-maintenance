import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, ctx
import plotly.graph_objects as go
from google.cloud import bigquery
import shap

# 1. Setup
TABLE_ID = "virtual-metrics-501014-f4.cnc_production.predictions_log" 
bq_client = bigquery.Client(location="europe-west3") 

with open('models/xgboost_cnc_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(MODEL)
feature_cols = ['Air Temp', 'Proc Temp', 'Temp Diff', 'RPM', 
                'Torque', 'Tool Wear', 'High RPM', 'High Tool Wear']

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    html.H1("CNC Predictive Maintenance Dashboard", className="text-center my-4"),
    dbc.Button("Generate New Predictions", id="btn-generate", color="primary", className="mb-3"),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("Total Predictions"), html.P(id="total-preds")])])),
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("System Status"), html.Div(id="status-indicator")])])),
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("Alert"), html.Div(id="alert-area")])])),
    ], className="mb-4"),
    dcc.Graph(id="live-graph"),
    html.H4("Why this prediction?", className="mt-4"),
    dcc.Graph(id="shap-graph"),
    dcc.Interval(id="interval-component", interval=5000, n_intervals=0)
])

# 3. Updated Backend Logic
def generate_and_upload(num_rows):
    try:
        df_source = pd.read_csv('data/ai4i2020.csv')
        # Drop rows with missing values to avoid NULLs in BigQuery
        clean_df = df_source.dropna(subset=['Air temperature [K]', 'Process temperature [K]', 
                                            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
        sample = clean_df.sample(n=num_rows)
        
        rows = []
        # Start from now, but space each prediction out by 3 seconds
        start_time = pd.Timestamp.now('UTC')
        
        for i, (_, row) in enumerate(sample.iterrows()):
            air_t = float(row["Air temperature [K]"] - 273.15)
            proc_t = float(row["Process temperature [K]"] - 273.15)
            torque = float(row["Torque [Nm]"])
            tool_wear = float(row["Tool wear [min]"])
            rpm = float(row["Rotational speed [rpm]"])
            
            features = np.array([[air_t, proc_t, round(proc_t - air_t, 2), rpm, torque, tool_wear, 
                                  1.0 if rpm > 2500 else 0.0, 1.0 if tool_wear >= 200 else 0.0]])
            prob = float(MODEL.predict_proba(features)[0][1])
            
            # Use 3-second spacing to prevent "clumping" on the X-axis
            timestamp = start_time + pd.Timedelta(seconds=i * 3)
            
            rows.append({
                "timestamp_utc": timestamp.isoformat(),
                "ml_failure_probability": round(prob, 4),
                "torque_nm": torque,
                "tool_wear_min": tool_wear
            })
            
        if rows:
            bq_client.insert_rows_json(TABLE_ID, rows)
            
    except Exception as e:
        print(f"DEBUG: Generation failed: {e}")

@app.callback(
    [Output("live-graph", "figure"), Output("total-preds", "children"),
     Output("status-indicator", "children"), Output("alert-area", "children"),
     Output("shap-graph", "figure")],
    [Input("btn-generate", "n_clicks"), Input("interval-component", "n_intervals")]
)
def update_dashboard(n_clicks, n_intervals):
    if ctx.triggered_id == "btn-generate" and n_clicks:
        generate_and_upload(num_rows=10)
    
    # YOUR ORIGINAL QUERY RE-INTEGRATED
    query = f"""
        SELECT timestamp_utc, ml_failure_probability, torque_nm, tool_wear_min, air_temp, proc_temp, temp_diff, rpm,
        CASE WHEN torque_nm > 60 THEN 'High Torque' WHEN tool_wear_min > 200 THEN 'High Tool Wear' ELSE 'Normal' END as alert_reason
        FROM `{TABLE_ID}`
        ORDER BY timestamp_utc DESC
        LIMIT 100
    """
    df = bq_client.query(query).to_dataframe()
    if df.empty: return go.Figure(), 0, "N/A", "Waiting...", go.Figure()
    
    df_plot = df.sort_values('timestamp_utc')
    fig = go.Figure(go.Scatter(x=df_plot['timestamp_utc'], y=df_plot['ml_failure_probability'], mode='lines'))
    fig.update_layout(title="Failure Probability", margin=dict(l=20, r=20, t=40, b=20))
    
    # SHAP & Status Logic
    latest = df.iloc[0]
    feat_vals = np.array([[latest['air_temp'], latest['proc_temp'], latest['temp_diff'], 
                           latest['rpm'], latest['torque_nm'], latest['tool_wear_min'], 
                           1.0 if latest['rpm'] > 2500 else 0.0, 1.0 if latest['tool_wear_min'] >= 200 else 0.0]])
    
    shap_vals = explainer.shap_values(feat_vals)[0]
    fig_shap = go.Figure([go.Bar(x=shap_vals, y=feature_cols, orientation='h', 
                                 marker=dict(color=['red' if x > 0 else 'blue' for x in shap_vals]))])
    fig_shap.update_layout(title="Feature Impact (Red=Risk, Blue=Stable)")

    latest_prob = latest['ml_failure_probability']
    status = dbc.Badge("CRITICAL", color="danger") if latest_prob > 0.7 else (
             dbc.Badge("WARNING", color="warning") if latest_prob > 0.5 else dbc.Badge("HEALTHY", color="success"))
    alert = dbc.Alert(f"{latest['alert_reason']}: {latest_prob:.2f}", color="danger" if latest_prob > 0.7 else "warning") \
            if latest_prob > 0.5 else dbc.Alert("System Stable", color="success")

    return fig, len(df), status, alert, fig_shap

if __name__ == "__main__":
    app.run_server(debug=True)