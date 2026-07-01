import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, ctx
from google.cloud import bigquery

# 1. Setup
TABLE_ID = "virtual-metrics-501014-f4.cnc_production.predictions_log" 
bq_client = bigquery.Client(location="europe-west3")

with open('models/xgboost_cnc_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# 2. Layout
metric_cards = dbc.Row([
    dbc.Col(dbc.Card([dbc.CardBody([html.H4("Total Predictions", className="card-title"), html.P(id="total-preds", className="card-text")])]), width=4),
    dbc.Col(dbc.Card([dbc.CardBody([html.H4("System Status", className="card-title"), html.Div(id="status-indicator")])]), width=4),
    dbc.Col(dbc.Card([dbc.CardBody([html.H4("Alert", className="card-title"), html.Div(id="alert-area")])]), width=4),
])

app.layout = dbc.Container([
    html.H1("CNC Predictive Maintenance Dashboard", className="text-center my-4"),
    dbc.Button("Generate & Predict Live Data", id="btn-generate", color="primary", className="mb-3"),
    metric_cards,
    dcc.Graph(id="live-graph"),
    dcc.Interval(id="interval-component", interval=60*1000, n_intervals=0)
])

# 3. Backend Logic
def generate_and_upload(num_rows):
    try:
        df_source = pd.read_csv('data/ai4i2020.csv')
        sample = df_source.sample(n=num_rows)
        rows = []
        for _, row in sample.iterrows():
            air_t, proc_t = round(row["Air temperature [K]"]-273.15, 2), round(row["Process temperature [K]"]-273.15, 2)
            features = np.array([[air_t, proc_t, round(proc_t-air_t, 2), row["Rotational speed [rpm]"], 
                                  row["Torque [Nm]"], row["Tool wear [min]"], 
                                  1.0 if row["Rotational speed [rpm]"] > 2500 else 0.0,
                                  1.0 if row["Tool wear [min]"] >= 200 else 0.0]])
            prob = float(MODEL.predict_proba(features)[0][1])
            rows.append({
                "timestamp_utc": pd.Timestamp.now('UTC').isoformat(),
                "ml_failure_probability": round(prob, 4),
                "torque_nm": float(row["Torque [Nm]"]),
                "tool_wear_min": float(row["Tool wear [min]"])
            })
        bq_client.insert_rows_json(TABLE_ID, rows)
    except Exception as e:
        print(f"DEBUG: Generation failed: {e}")

@app.callback(
    [Output("live-graph", "figure"), Output("total-preds", "children"),
     Output("status-indicator", "children"), Output("alert-area", "children")],
    [Input("btn-generate", "n_clicks"), Input("interval-component", "n_intervals")]
)
def update_dashboard(n_clicks, n_intervals):
    # Trigger generation
    if ctx.triggered_id == "btn-generate" and n_clicks:
        generate_and_upload(num_rows=5)
    
    query = f"""
        SELECT timestamp_utc, ml_failure_probability, torque_nm, tool_wear_min,
        CASE WHEN torque_nm > 60 THEN 'High Torque' WHEN tool_wear_min > 200 THEN 'High Tool Wear' ELSE 'Normal' END as alert_reason
        FROM `{TABLE_ID}` WHERE timestamp_utc >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        ORDER BY timestamp_utc DESC LIMIT 500
    """
    df = bq_client.query(query).to_dataframe()
    
    # Empty state handling
    if df.empty:
        return px.line(title="No Data Available"), 0, "N/A", "Waiting for data..."
    
    fig = px.line(df, x="timestamp_utc", y="ml_failure_probability", title="Recent Failure Predictions")
    status = dbc.Badge("HEALTHY", color="success") if df['ml_failure_probability'].mean() < 0.5 else dbc.Badge("WARNING", color="warning")
    alert = dbc.Alert(f"CRITICAL: {df['alert_reason'].iloc[0]}", color="danger") if df['ml_failure_probability'].iloc[0] > 0.7 else dbc.Alert("System Stable", color="success")
    
    return fig, len(df), status, alert

if __name__ == "__main__":
    app.run_server(debug=True)