import pandas as pd
import numpy as np
from datetime import datetime

def extract_features_advanced(borrower_id: int, payment_df: pd.DataFrame):
    """
    Extracts advanced features and a repayment summary for a single borrower.
    Returns two dictionaries: one for the model features, one for the summary.
    """
    borrower_history = payment_df[payment_df['borrower_id'] == borrower_id].copy()
    borrower_history.sort_values(by='due_date', inplace=True)

    # --- Handle Cold Start (Borrower with no history) ---
    if borrower_history.empty:
        summary = {
            "total_payments": 0, "on_time": 0, "missed": 0, "due": 0,
            "average_delay_days": 0.0, "max_delay_days": 0
        }
        # Return default features (e.g., zeros) for the model
        features = {
            'on_time_ratio': 0.0, 'missed_ratio': 0.0, 'due_ratio': 0.0, 'average_delay': 0.0,
            'max_delay': 0.0, 'std_dev_delay': 0.0, 'max_missed_streak': 0,
            'recent_missed_count': 0, 'recent_due_count': 0, 'recent_on_time_count': 0,
            'time_since_last_missed': 9999, 'time_since_last_due': 9999
        }
        return features, summary

    # --- Basic Counts and Ratios ---
    status_counts = borrower_history['status'].value_counts()
    on_time_count = status_counts.get('on_time', 0)
    missed_count = status_counts.get('missed', 0)
    due_count = status_counts.get('due', 0)
    total_payments = len(borrower_history)

    on_time_ratio = on_time_count / total_payments if total_payments > 0 else 0
    missed_ratio = missed_count / total_payments if total_payments > 0 else 0
    due_ratio = due_count / total_payments if total_payments > 0 else 0

    # --- Delay Calculations (only for 'missed' payments) ---
    missed_payments = borrower_history[borrower_history['status'] == 'missed'].copy()
    if not missed_payments.empty:
        missed_payments['delay_days'] = (missed_payments['payment_date'] - missed_payments['due_date']).dt.days
        average_delay = missed_payments['delay_days'].mean()
        max_delay = missed_payments['delay_days'].max()
        std_dev_delay = missed_payments['delay_days'].std()
    else:
        average_delay, max_delay, std_dev_delay = 0.0, 0.0, 0.0

    # --- Streak Calculation ---
    max_missed_streak = 0
    current_streak = 0
    for status in borrower_history['status']:
        if status == 'missed':
            current_streak += 1
        else:
            max_missed_streak = max(max_missed_streak, current_streak)
            current_streak = 0
    max_missed_streak = max(max_missed_streak, current_streak)

    # --- Recent Behavior (last 5 payments) ---
    recent_history = borrower_history.tail(5)
    recent_status_counts = recent_history['status'].value_counts()
    recent_missed_count = recent_status_counts.get('missed', 0)
    recent_due_count = recent_status_counts.get('due', 0)
    recent_on_time_count = recent_status_counts.get('on_time', 0)
    
    # --- Time Since Last Event ---
    now = datetime.now()
    last_missed_date = borrower_history[borrower_history['status'] == 'missed']['due_date'].max()
    last_due_date = borrower_history[borrower_history['status'] == 'due']['due_date'].max()

    time_since_last_missed = (now - last_missed_date).days if pd.notna(last_missed_date) else 9999
    time_since_last_due = (now - last_due_date).days if pd.notna(last_due_date) else 9999

    # --- Assemble Dictionaries ---
    features = {
        'on_time_ratio': on_time_ratio, 'missed_ratio': missed_ratio, 'due_ratio': due_ratio,
        'average_delay': average_delay, 'max_delay': max_delay, 'std_dev_delay': std_dev_delay,
        'max_missed_streak': max_missed_streak, 'recent_missed_count': recent_missed_count,
        'recent_due_count': recent_due_count, 'recent_on_time_count': recent_on_time_count,
        'time_since_last_missed': time_since_last_missed, 'time_since_last_due': time_since_last_due
    }
    # Fill any potential NaNs from std dev calculation
    features = {k: np.nan_to_num(v) for k, v in features.items()}

    summary = {
        "total_payments": total_payments,
        "on_time": on_time_count,
        "missed": missed_count,
        "due": due_count,
        "average_delay_days": round(average_delay, 2),
        "max_delay_days": int(max_delay)
    }

    return features, summary