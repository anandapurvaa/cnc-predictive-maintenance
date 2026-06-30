import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import pickle
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access the variables
db_path = os.getenv("DB_PATH")
broker_url = os.getenv("BROKER_URL")

print(f"Connecting to: {broker_url}")

def train_predictive_model():
    # Find raw data relative to this script
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ai4i2020.csv')
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Could not find raw data file at {DATA_PATH}")
        return

    print("📥 Loading full baseline dataset for high-fidelity training...")
    df_raw = pd.read_csv(DATA_PATH)
    
    # Process the raw data to perfectly match our production warehouse schema
    df = pd.DataFrame()
    df['air_temperature_c'] = round(df_raw["Air temperature [K]"] - 273.15, 2)
    df['process_temperature_c'] = round(df_raw["Process temperature [K]"] - 273.15, 2)
    df['temperature_differential_c'] = round(df['process_temperature_c'] - df['air_temperature_c'], 2)
    df['rotational_speed_rpm'] = df_raw["Rotational speed [rpm]"]
    df['torque_nm'] = df_raw["Torque [Nm]"]
    df['tool_wear_min'] = df_raw["Tool wear [min]"]
    
    # Feature Engineering matching dbt flags
    df['is_high_speed_anomalous'] = np.where(df['rotational_speed_rpm'] > 2500, 1.0, 0.0)
    df['is_critical_wear_risk'] = np.where(df['tool_wear_min'] >= 200, 1.0, 0.0)
    
    # Target
    df['has_failed'] = df_raw["Machine failure"]

    print(f"✅ Successfully prepared {len(df)} records ({int(df['has_failed'].sum())} actual failures included).")
    
    # Split features and target
    X = df.drop(columns=['has_failed'])
    y = df['has_failed']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🤖 Training robust XGBoost Predictive Maintenance Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        scale_pos_weight=15 # Explicitly handles rare failure events
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    predictions = model.predict(X_test)
    prob_predictions = model.predict_proba(X_test)[:, 1]
    
    print("\n指标 Model Performance Report:")
    print(classification_report(y_test, predictions))
    print(f"ROC AUC Score: {round(roc_auc_score(y_test, prob_predictions), 4)}")
    
    # Save artifact
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_output_path = os.path.join(models_dir, 'xgboost_cnc_model.pkl')
    
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Robust model artifact saved to: {model_output_path}")

if __name__ == "__main__":
    train_predictive_model()