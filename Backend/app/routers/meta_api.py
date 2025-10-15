# app/routers/meta_api.py
from fastapi import APIRouter
import json
from pathlib import Path
from ..deps import MODELS_DIR

router = APIRouter(prefix="/model", tags=["meta"])

@router.get("/info")
def info():
    out = {}
    for name in ["meta_GradientBoosting_yield.json","meta_croprec.json","meta_cluster.json"]:
        p = MODELS_DIR / name
        out[name] = json.loads(p.read_text()) if p.exists() else {"status":"missing"}
    return out
