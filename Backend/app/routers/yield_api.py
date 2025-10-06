# app/routers/yield_api.py
from fastapi import APIRouter, HTTPException
import pandas as pd
from ..schemas import YieldPredictionRequest, YieldPredictionResponse, YieldPredictionItem
from ..deps import load_artifact

router = APIRouter(prefix="/predict", tags=["yield"])

model = None

@router.on_event("startup")
def _load():
    global model
    # model = load_artifact("yield_model.joblib")
    model = load_artifact("GradientBoosting_yield_model.joblib")

@router.post("/yield", response_model=YieldPredictionResponse)
def predict_yield(req: YieldPredictionRequest):
    try:
        rows = [s.normalized_dict() for s in req.samples]
        df = pd.DataFrame(rows)[
            ["N","P","K","pH","rainfall","temperature","State_Name","Season","Crop"]
        ]
        preds = model.predict(df)
        return YieldPredictionResponse(
            predictions=[YieldPredictionItem(predicted_yield_ton_per_hec=float(x)) for x in preds]
        )
    except Exception as e:
        raise HTTPException(400, f"Prediction error: {e}")
