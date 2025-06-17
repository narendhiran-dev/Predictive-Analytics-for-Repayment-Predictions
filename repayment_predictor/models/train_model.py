# =============================================================================
# START: Direct Path Configuration for Your Machine
# This block uses your specific folder path to ensure Python finds the modules.
# =============================================================================
import sys
import pathlib

# Directly setting the project root to your specific folder path.
# NOTE: If you ever move this project folder, you must update this line.
your_project_path = "C:/Users/devid/Downloads/repayment_prediction_system"
project_root = pathlib.Path(your_project_path)

# Add this path to the system so Python can find the 'repayment_predictor' module.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project Root has been explicitly set to: {project_root}")
# =============================================================================
# END: Direct Path Configuration
# =============================================================================


# Now, the rest of your imports will work perfectly.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# These imports now work because of the path configuration above
from repayment_predictor.core import config
from repayment_predictor.data_processing import loader
from repayment_predictor.features import build_features


def run_training():
    """Trains the model and saves the artifacts."""
    print("\n--- Starting Model Training ---")

    # 1. Load Data
    print("Step 1: Loading data...")
    payment_df, investor_df = loader.load_payment_data()
    if payment_df is None:
        print("Data loading failed. Exiting.")
        return
    print("Data loaded successfully.")

    # 2. Feature Engineering for all borrowers
    print("Step 2: Building features for all borrowers...")
    all_borrower_ids = payment_df['borrower_id'].unique()
    feature_list = []
    for borrower_id in all_borrower_ids:
        features, _ = build_features.extract_features_advanced(borrower_id, payment_df)
        features['borrower_id'] = borrower_id
        feature_list.append(features)
    
    feature_df = pd.DataFrame(feature_list)
    print(f"Features built for {len(feature_df)} borrowers.")

    # 3. Create Target Variable (Synthetic Target)
    print("Step 3: Creating synthetic target variable...")
    np.random.seed(42)
    feature_df['target'] = feature_df['on_time_ratio'] * 100 + np.random.normal(0, 2, size=len(feature_df))
    feature_df['target'] = np.clip(feature_df['target'], 0, 100)

    # 4. Data Splitting
    print("Step 4: Splitting data into training and testing sets...")
    X = feature_df[config.FEATURES]
    y = feature_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Data Scaling
    print("Step 5: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Model Training
    print("Step 6: Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    model.fit(X_train_scaled, y_train)

    # 7. Evaluation
    print("Step 7: Evaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Training Complete ---")
    print(f"Mean Squared Error (Test Set): {mse:.2f}")
    print(f"R-squared (Test Set): {r2:.2f}")
    print(f"Out-of-Bag Score: {model.oob_score_:.2f}")

    # 8. Save Artifacts
    print("Step 8: Saving model and scaler...")
    config.SAVED_MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"  -> Model saved to: {config.MODEL_PATH}")
    print(f"  -> Scaler saved to: {config.SCALER_PATH}")
    print("\n--- Process Finished ---")


if __name__ == "__main__":
    run_training()