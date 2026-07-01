import dash 
from dash import dcc, html
import plotly.express as px
from google.cloud import bigquery
import pandas as pd
from dotenv import load_dotenv
import os
from flask_caching import Cache

# Add this to load your .env settings
load_dotenv()
app = dash.Dash(__name__)
cache = Cache(app.server, config={'CACHE_TYPE': 'SimpleCache'})

# 1. Initialize BQ Client
bq_client = bigquery.Client()
TABLE_ID = "virtual-metrics-501014-f4.cnc_production.predictions_log"

@cache.memoize(timeout=60) # Cache the result for 60 seconds
def fetch_data():
    query = f"SELECT * FROM `{TABLE_ID}` ORDER BY timestamp_utc DESC LIMIT 500"
    return bq_client.query(query).to_dataframe()

# 2. Build the Dashboard App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("CNC Predictive Maintenance: Model Health Dashboard"),
    
    # Adding a simple KPI card
    html.Div(id='live-update-text', style={'fontSize': 24, 'fontWeight': 'bold', 'color': 'red'}),
    
    dcc.Interval(id='graph-update', interval=5*1000, n_intervals=0), # Refreshed to 5s
    dcc.Graph(id='live-graph')
])

@app.callback(
    dash.Output('live-graph', 'figure'),
    [dash.Input('graph-update', 'n_intervals')]
)
@app.callback(
    dash.Output('live-graph', 'figure'),
    [dash.Input('graph-update', 'n_intervals')]
)
def update_graph(n):
    df = fetch_data()
    
    # Create the line chart
    fig = px.line(df, x='timestamp_utc', y='ml_failure_probability', 
                  title='Model Risk Probability Over Time')
    
    # Add a horizontal threshold line at 0.8 (Critical Risk)
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                  annotation_text="Critical Threshold")
    
    # Style the graph to be more readable
    fig.update_layout(
        yaxis_range=[0, 1.1], # Keeps the scale fixed between 0 and 1
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

server = app.server

if __name__ == '__main__':
    # Get the port from the environment, default to 8080 for Cloud Run
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)