from repayment_predictor.core import config

def classify_risk(predicted_percentage: float) -> str:
    """Classifies risk level based on the predicted repayment percentage."""
    if predicted_percentage < config.RISK_THRESHOLDS["HIGH"]:
        return "High Risk"
    elif predicted_percentage < config.RISK_THRESHOLDS["MEDIUM"]:
        return "Medium Risk"
    else:
        return "Low Risk"