import os
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import pickle

# Setup GCP configuration
KEY_PATH = os.path.join(os.path.dirname(__file__), '..', 'gcp-key.json')
if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

def train_predictive_model():
    # Initialize BigQuery Client
    client = bigquery.Client()
    
    # Target our dbt feature engineered table
    query = """
        SELECT 
            air_temperature_c,
            process_temperature_c,
            temperature_differential_c,
            rotational_speed_rpm,
            torque_nm,
            tool_wear_min,
            is_high_speed_anomalous,
            is_critical_wear_risk,
            has_failed
        FROM `virtual-metrics-501014-f4.raw_factory_data.fct_cnc_failures`
    """
    
    print("📥 Pulling feature-engineered analytics data from BigQuery...")
    df = client.query(query).to_dataframe()
    
    if df.empty:
        print("❌ No data found in the analytical table. Ensure your simulator ran and dbt executed successfully.")
        return

    print(f"✅ Successfully loaded {len(df)} records from the cloud data warehouse.")
    
    # Split features and target
    X = df.drop(columns=['has_failed'])
    y = df['has_failed']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🤖 Training XGBoost Predictive Maintenance Classifier...")
    # Initialize and fit the model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=10 # Handles class imbalance (failures are rare)
    )
    model.fit(X_train, y_train)
    
    # Model Evaluation
    predictions = model.predict(X_test)
    prob_predictions = model.predict_proba(X_test)[:, 1]
    
    print("\n📊 Model Performance Report:")
    print(classification_report(y_test, predictions))
    print(f"ROC AUC Score: {round(roc_auc_score(y_test, prob_predictions), 4)}")
    
    # Save the trained model artifact
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_output_path = os.path.join(models_dir, 'xgboost_cnc_model.pkl')
    
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Model artifact successfully saved locally to: {model_output_path}")

if __name__ == "__main__":
    train_predictive_model()