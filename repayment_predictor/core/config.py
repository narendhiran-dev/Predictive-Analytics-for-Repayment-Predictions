# repayment_predictor/core/config.py

import pathlib

# This is the updated, dynamic way to find the project root.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# --- PATHS ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

PAYMENT_HISTORY_PATH = RAW_DATA_DIR / "payment_history.csv"
INVESTOR_BORROWER_PATH = RAW_DATA_DIR / "investor_borrower.csv"

MODEL_NAME = "random_forest_regressor.joblib"
SCALER_NAME = "scaler.joblib"
MODEL_PATH = SAVED_MODELS_DIR / MODEL_NAME
SCALER_PATH = SAVED_MODELS_DIR / SCALER_NAME


# --- THIS IS THE FIX ---
# The feature names here now exactly match the ones created in build_features.py
FEATURES = [
    'on_time_ratio', 'missed_ratio', 'due_ratio',
    'average_delay',          # Corrected name
    'max_delay',              # Corrected name
    'std_dev_delay',          # Corrected name
    'max_missed_streak',
    'recent_missed_count', 'recent_due_count', 'recent_on_time_count',
    'time_since_last_missed', # Corrected name
    'time_since_last_due'     # Corrected name
]


# --- RISK CLASSIFICATION ---
RISK_THRESHOLDS = {
    "HIGH": 70.0,
    "MEDIUM": 90.0
}