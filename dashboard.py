import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, ctx
import plotly.graph_objects as go
from google.cloud import bigquery

# 1. Setup - Ensure location matches your dataset!
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
    dbc.Button("Generate New Predictions", id="btn-generate", color="primary", className="mb-3"),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("Total Predictions", className="card-title"), html.P(id="total-preds", className="card-text")])])),
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("System Status", className="card-title"), html.Div(id="status-indicator")])])),
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("Alert", className="card-title"), html.Div(id="alert-area")])])),
    ]),
    dcc.Graph(id="live-graph"),
    dcc.Interval(id="interval-component", interval=60*1000, n_intervals=0)
])

# 3. Backend Logic
# 3. Updated Backend Logic
def generate_and_upload(num_rows):
    try:
        df_source = pd.read_csv('data/ai4i2020.csv')
        # Ensure we only pick rows that have all necessary data
        clean_df = df_source.dropna(subset=['Air temperature [K]', 'Process temperature [K]', 
                                            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
        sample = clean_df.sample(n=num_rows)
        
        rows = []
        for _, row in sample.iterrows():
            # 1. Extract and validate features
            air_t = float(row["Air temperature [K]"] - 273.15)
            proc_t = float(row["Process temperature [K]"] - 273.15)
            torque = float(row["Torque [Nm]"])
            tool_wear = float(row["Tool wear [min]"])
            rpm = float(row["Rotational speed [rpm]"])
            
            features = np.array([[air_t, proc_t, round(proc_t - air_t, 2), rpm, 
                                  torque, tool_wear, 
                                  1.0 if rpm > 2500 else 0.0,
                                  1.0 if tool_wear >= 200 else 0.0]])
            
            prob = float(MODEL.predict_proba(features)[0][1])
            
            # 2. Append only if we have valid data
            rows.append({
                "timestamp_utc": pd.Timestamp.now('UTC').isoformat(),
                "ml_failure_probability": round(prob, 4),
                "torque_nm": torque,
                "tool_wear_min": tool_wear
            })
            
        if rows:
            bq_client.insert_rows_json(TABLE_ID, rows)
            print(f"DEBUG: Successfully inserted {len(rows)} rows.")
            
    except Exception as e:
        print(f"DEBUG: Generation failed: {str(e)}")

@app.callback(
    [Output("live-graph", "figure"), Output("total-preds", "children"),
     Output("status-indicator", "children"), Output("alert-area", "children")],
    [Input("btn-generate", "n_clicks"), Input("interval-component", "n_intervals")]
)
def update_dashboard(n_clicks, n_intervals):
    # Trigger generation
    if ctx.triggered_id == "btn-generate" and n_clicks:
        # Simplified: removed the 5-min offset to use actual system time
        generate_and_upload(num_rows=5)
    
    query = f"""
        SELECT timestamp_utc, ml_failure_probability, torque_nm, tool_wear_min,
        CASE WHEN torque_nm > 60 THEN 'High Torque' WHEN tool_wear_min > 200 THEN 'High Tool Wear' ELSE 'Normal' END as alert_reason
        FROM `{TABLE_ID}` WHERE timestamp_utc >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        ORDER BY timestamp_utc DESC LIMIT 500
    """
    df = bq_client.query(query).to_dataframe().sort_values('timestamp_utc')
    
    if df.empty:
        return go.Figure(), 0, "N/A", "Waiting..."
    
    # Create plot with Critical Threshold Line
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp_utc'], y=df['ml_failure_probability'], mode='lines', name='Failure Prob'))
    
    # Adding the Critical Threshold Line
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Critical Threshold (0.7)")
    
    fig.update_layout(title="Failure Probability Over Time", xaxis_title="Timestamp (UTC)", yaxis_title="Probability")
    
    status = dbc.Badge("HEALTHY", color="success") if df['ml_failure_probability'].mean() < 0.5 else dbc.Badge("WARNING", color="warning")
    alert = dbc.Alert(f"CRITICAL: {df['alert_reason'].iloc[-1]}", color="danger") if df['ml_failure_probability'].iloc[-1] > 0.7 else dbc.Alert("System Stable", color="success")
    
    return fig, len(df), status, alert

if __name__ == "__main__":
    app.run_server(debug=True)