# app/routers/cluster_api.py
from fastapi import APIRouter, HTTPException
import pandas as pd
from ..schemas import ClusterAssignRequest, ClusterAssignResponse, ClusterAssignItem
from ..deps import load_artifact

router = APIRouter(prefix="/cluster", tags=["cluster"])

pipe = None  # scaler + kmeans
cluster_profile = None  # optional: per-cluster means

@router.on_event("startup")
def _load():
    global pipe, cluster_profile
    pipe = load_artifact("cluster_model.joblib")
    # try to load cluster profile (per-cluster means) if baked-in
    try:
        cluster_profile = load_artifact("cluster_profile.joblib")
    except Exception:
        cluster_profile = None

@router.post("/assign", response_model=ClusterAssignResponse)
def assign_cluster(req: ClusterAssignRequest):
    try:
        df = pd.DataFrame([s.dict() for s in req.samples])[["N","P","K","pH","rainfall","temperature"]]
        cluster_ids = pipe.predict(df).tolist()
        return ClusterAssignResponse(assignments=[ClusterAssignItem(cluster_id=int(c)) for c in cluster_ids])
    except Exception as e:
        raise HTTPException(400, f"Clustering error: {e}")
