"""FastAPI inference service for the frozen Week 7 Telco churn pipeline."""
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
THRESHOLD = 0.3251224160194397
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

class CustomerFeatures(BaseModel):
    """Raw customer fields expected by the Week 7 sklearn Pipeline."""
    model_config = ConfigDict(extra="forbid")
    gender: str = Field(min_length=1)
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str = Field(min_length=1)
    Dependents: str = Field(min_length=1)
    tenure: int = Field(ge=0, le=72)
    PhoneService: str = Field(min_length=1)
    MultipleLines: str = Field(min_length=1)
    InternetService: str = Field(min_length=1)
    OnlineSecurity: str = Field(min_length=1)
    OnlineBackup: str = Field(min_length=1)
    DeviceProtection: str = Field(min_length=1)
    TechSupport: str = Field(min_length=1)
    StreamingTV: str = Field(min_length=1)
    StreamingMovies: str = Field(min_length=1)
    Contract: str = Field(min_length=1)
    PaperlessBilling: str = Field(min_length=1)
    PaymentMethod: str = Field(min_length=1)
    MonthlyCharges: float = Field(ge=0, le=200)
    TotalCharges: float = Field(ge=0, le=10000)

    @field_validator(
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    )
    @classmethod
    def strip_category(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("category must not be blank")
        # Unknown non-empty categories are intentionally allowed:
        # OneHotEncoder(handle_unknown='ignore') handles them safely.
        return value

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str
    threshold: float

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    version="1.0.0",
    description="Serves the frozen Week 7 preprocessing + XGBoost pipeline.",
)

_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
        # Only load artifacts produced by this trusted project.
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "week7-xgboost-pipeline"}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    try:
        row = pd.DataFrame([customer.model_dump()], columns=FEATURE_COLUMNS)
        probability = float(get_model().predict_proba(row)[0, 1])
        label = "Yes" if probability >= THRESHOLD else "No"
        return PredictionResponse(
            churn_probability=probability,
            prediction=label,
            threshold=THRESHOLD,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
