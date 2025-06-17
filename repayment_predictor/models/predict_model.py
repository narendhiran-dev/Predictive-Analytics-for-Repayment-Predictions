# repayment_predictor/models/predict_model.py

import sys
import pathlib
import pandas as pd
import joblib

# --- Self-Correcting Path Magic ---
current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Magic ---

from repayment_predictor.core import config
from repayment_predictor.data_processing import loader
from repayment_predictor.features import build_features
from repayment_predictor.models import risk_classifier

# --- Load artifacts once at startup ---
model = None
scaler = None
payment_df = None
investor_df = None

try:
    print("Loading model artifacts...")
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    payment_df, investor_df = loader.load_payment_data()
    print("Model artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not load model or data: {e}. Please train the model first.")
except Exception as e:
    print(f"An unexpected error occurred during artifact loading: {e}")


def get_prediction(borrower_id: int):
    """Generates a repayment prediction for a given borrower."""
    if model is None or payment_df is None or investor_df is None:
        return {"error": "Model or data not loaded. The server is not ready."}

    # Check if borrower exists in any loan
    if borrower_id not in payment_df['borrower_id'].unique() and borrower_id not in investor_df['borrower_id'].unique():
        return {"error": f"Borrower with ID {borrower_id} not found."}

    # 1. Extract features and summary
    features_dict, summary_dict = build_features.extract_features_advanced(borrower_id, payment_df)

    # 2. Handle cold start case (new borrower with no payment history)
    if summary_dict['total_payments'] == 0:
        prediction = 75.0  # Default prediction for new borrowers
    else:
        # 3. Prepare features for prediction
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[config.FEATURES] # Ensure column order
        scaled_features = scaler.transform(features_df)
        
        # 4. Make prediction
        prediction = model.predict(scaled_features)[0]
        prediction = round(float(prediction), 2)
        prediction = max(0, min(100, prediction))

    # 5. Classify risk
    risk = risk_classifier.classify_risk(prediction)

    return {
        "borrower_repayment_summary": summary_dict,
        "predicted_repayment_percentage": prediction,
        "risk_level": risk
    }