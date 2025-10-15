# app/schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field, confloat, constr

# Basic bounds (adjust if you tighten domain rules)
PH = confloat(ge=0, le=14)
Temp = confloat(ge=-20, le=60)
Rain = confloat(ge=0)
NPK = confloat(ge=0)
StrNorm = constr(strip_whitespace=True, min_length=1)

class BaseAgriFeatures(BaseModel):
    State_Name: StrNorm
    Season: StrNorm
    Crop: StrNorm
    N: NPK
    P: NPK
    K: NPK
    pH: PH
    rainfall: Rain
    temperature: Temp

    # normalize to match training (lowercase/trim)
    def normalized_dict(self):
        d = self.dict()
        for k in ["State_Name","Season","Crop"]:
            d[k] = d[k].strip().lower()
        return d

# ---- Yield Prediction
class YieldPredictionRequest(BaseModel):
    samples: List[BaseAgriFeatures] = Field(..., min_items=1)

class YieldPredictionItem(BaseModel):
    predicted_yield_ton_per_hec: float

class YieldPredictionResponse(BaseModel):
    predictions: List[YieldPredictionItem]
    model: Optional[str] = "gradient_boosting"
    version: Optional[str] = "1.0.0"

# ---- Crop Recommendation
class CropRecFeatures(BaseModel):
    # same schema but Crop is not required as an input (we predict Crop)
    State_Name: StrNorm
    Season: StrNorm
    N: NPK
    P: NPK
    K: NPK
    pH: PH
    rainfall: Rain
    temperature: Temp

    def normalized_dict(self):
        d = self.dict()
        for k in ["State_Name","Season"]:
            d[k] = d[k].strip().lower()
        return d

class CropRecommendationRequest(BaseModel):
    samples: List[CropRecFeatures] = Field(..., min_items=1)

class CropRecItem(BaseModel):
    top_3_crops: List[str]
    confidence_scores: Optional[List[float]] = None  # Confidence for each crop

class CropRecommendationResponse(BaseModel):
    recommendations: List[CropRecItem]
    model: Optional[str] = "knn"
    version: Optional[str] = "1.0.0"

# ---- Clustering (agro-regimes)
class ClusterAssignFeatures(BaseModel):
    N: NPK
    P: NPK
    K: NPK
    pH: PH
    rainfall: Rain
    temperature: Temp

class ClusterAssignRequest(BaseModel):
    samples: List[ClusterAssignFeatures] = Field(..., min_items=1)

class ClusterAssignItem(BaseModel):
    cluster_id: int

class ClusterAssignResponse(BaseModel):
    assignments: List[ClusterAssignItem]
    model: Optional[str] = "kmeans"
    version: Optional[str] = "1.0.0"
