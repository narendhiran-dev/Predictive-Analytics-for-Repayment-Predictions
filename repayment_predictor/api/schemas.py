from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    investor_id: str
    borrower_id: int

class BorrowerSummary(BaseModel):
    total_payments: int
    on_time: int
    missed: int
    due: int
    average_delay_days: float
    max_delay_days: int

class PredictionResponse(BaseModel):
    borrower_repayment_summary: BorrowerSummary
    predicted_repayment_percentage: float = Field(..., example=77.23)
    risk_level: str = Field(..., example="Medium Risk")