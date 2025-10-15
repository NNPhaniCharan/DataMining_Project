# app/routers/crop_api.py
from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from ..schemas import CropRecommendationRequest, CropRecommendationResponse, CropRecItem
from ..deps import load_artifact

router = APIRouter(prefix="/recommend", tags=["crop"])

pipe = None  # preprocessing+knn
label_classes = None  # classes_ to map indices back to crop names

@router.on_event("startup")
def _load():
    global pipe, label_classes
    pipe = load_artifact("croprec_model.joblib")
    # classes stored in pipe.named_steps["clf"].classes_
    label_classes = pipe.named_steps["clf"].classes_

@router.post("/crop", response_model=CropRecommendationResponse)
def recommend_crop(req: CropRecommendationRequest):
    try:
        X = pd.DataFrame([s.normalized_dict() for s in req.samples])[
            ["N","P","K","pH","rainfall","temperature","State_Name","Season"]
        ]
        # Get top 3 crop recommendations using kneighbors
        TOP_K = 3
        clf = pipe.named_steps["clf"]
        Xt = pipe.named_steps["pre"].transform(X)
        
        # Use more neighbors to ensure we get 3 distinct crops
        n_neighbors_to_check = min(100, len(label_classes))
        dist, idx = clf.kneighbors(Xt, n_neighbors=n_neighbors_to_check)
        
        # Convert neighbor indices to labels with frequency vote → top-3 labels by count
        recs = []
        for i, row_idx in enumerate(idx):
            labels = label_classes[clf._y[row_idx]]  # nearest neighbors' labels
            distances = dist[i]
            
            # Rank by frequency (with distance weighting)
            uniq, inverse = np.unique(labels, return_inverse=True)
            
            # Weight by inverse distance
            weights = 1 / (distances + 1e-10)
            
            # Sum weights for each unique crop
            crop_scores = np.zeros(len(uniq))
            for j, crop_idx in enumerate(inverse):
                crop_scores[crop_idx] += weights[j]
            
            # Sort by score (descending) and get top 3
            order = np.argsort(-crop_scores)
            top_crops = uniq[order][:TOP_K].tolist()
            top_scores = crop_scores[order][:TOP_K]
            
            # Normalize scores to percentages
            total_score = top_scores.sum()
            confidence_percentages = ((top_scores / total_score) * 100).tolist() if total_score > 0 else [0.0] * TOP_K
            
            # Ensure we have exactly 3 recommendations
            if len(top_crops) < TOP_K:
                remaining_crops = [c for c in label_classes if c not in top_crops]
                top_crops.extend(remaining_crops[:TOP_K - len(top_crops)])
                # Pad confidence scores with low values
                confidence_percentages.extend([0.0] * (TOP_K - len(confidence_percentages)))
            
            recs.append(CropRecItem(top_3_crops=top_crops, confidence_scores=confidence_percentages))
        return CropRecommendationResponse(recommendations=recs)
    except Exception as e:
        raise HTTPException(400, f"Recommendation error: {e}")
