# repayment_predictor/data_processing/loader.py

import pandas as pd
from repayment_predictor.core import config
import os # Import the os module

def load_payment_data():
    """Loads the payment history and investor-borrower mapping data."""
    try:
        # --- THIS IS THE FIX ---
        # We explicitly tell pandas to convert these two columns into datetime objects.
        payment_df = pd.read_csv(
            config.PAYMENT_HISTORY_PATH,
            parse_dates=['due_date', 'payment_date']
        )
        
        # Load investor-borrower mapping
        investor_df = pd.read_csv(config.INVESTOR_BORROWER_PATH)
        
        return payment_df, investor_df
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None