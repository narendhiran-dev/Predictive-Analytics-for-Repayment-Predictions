# repayment_predictor/api/main.py

import sys
import pathlib
from fastapi import FastAPI, HTTPException

# --- Self-Correcting Path Magic ---
current_file_path = pathlib.Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Path Magic ---

from repayment_predictor.api import schemas
from repayment_predictor.models import predict_model

app = FastAPI(
    title="Borrower Repayment Prediction API",
    description="Predicts a borrower's repayment behaviour based on past data.",
    version="1.0.0"
)

@app.post("/predict-repayment", response_model=schemas.PredictionResponse)
async def predict_repayment(request: schemas.PredictionRequest):
    """
    Predicts the repayment behaviour of a borrower.
    - **investor_id**: The ID of the investor (for logging/auth).
    - **borrower_id**: The ID of the borrower to predict for.
    """
    result = predict_model.get_prediction(borrower_id=request.borrower_id)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
        
    return result

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Repayment Prediction API is running and ready to make predictions."}