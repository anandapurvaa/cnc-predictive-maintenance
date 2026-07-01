from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import pickle
import numpy as np
from google.cloud import bigquery

# 1. Initialization
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
MODEL = pickle.load(open('models/xgboost_cnc_model.pkl', 'rb'))
bq_client = bigquery.Client()
TABLE_ID = "your-project-id.cnc_production.predictions_log"

# 2. Layout
app.layout = dbc.Container([
    html.H1("CNC Predictive Maintenance Dashboard", className="text-center my-4"),
    dbc.Button("Generate & Predict Live Data", id="btn-generate", color="primary", className="mb-3"),
    html.Div(id="refresh-status"),
    dcc.Graph(id="live-graph"),
    dcc.Interval(id="interval-component", interval=60*1000, n_intervals=0) # Auto-refresh every min
], fluid=True)

# 3. Logic: Data Generation
def generate_and_upload(num_rows):
    df_source = pd.read_csv('data/ai4i2020.csv')
    sample = df_source.sample(n=num_rows)
    
    rows_to_insert = []
    for _, row in sample.iterrows():
        # Example feature engineering (Update to match your training features!)
        features = np.array([[row["Air temperature [K]"], row["Process temperature [K]"], 
                              row["Rotational speed [rpm]"], row["Torque [Nm]"], 
                              row["Tool wear [min]"]]])
        
        prob = float(MODEL.predict_proba(features)[0][1])
        
        rows_to_insert.append({
            "timestamp_utc": pd.Timestamp.now('UTC').isoformat(),
            "machine_id": int(row["UDI"]),
            "ml_failure_probability": round(prob, 4)
        })
    
    bq_client.insert_rows_json(TABLE_ID, rows_to_insert)
    return f"Generated {num_rows} points."

# 4. Callbacks
@app.callback(
    [Output("live-graph", "figure"), Output("refresh-status", "children")],
    [Input("btn-generate", "n_clicks"), Input("interval-component", "n_intervals")]
)
def update_dashboard(n_clicks, n_intervals):
    # Trigger generation if button clicked
    status = ""
    if n_clicks and n_clicks > 0:
        status = generate_and_upload(5)
    
    # Query BigQuery for latest data
    query = f"SELECT * FROM `{TABLE_ID}` ORDER BY timestamp_utc DESC LIMIT 50"
    df = bq_client.query(query).to_dataframe()
    
    fig = px.line(df, x="timestamp_utc", y="ml_failure_probability", title="Recent Failure Predictions")
    return fig, status

server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)