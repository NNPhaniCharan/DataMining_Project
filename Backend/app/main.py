# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import yield_api, crop_api, cluster_api, meta_api, data_api, evaluation_api

app = FastAPI(title="DM Agri Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(yield_api.router)
app.include_router(crop_api.router)
app.include_router(cluster_api.router)
app.include_router(meta_api.router)
app.include_router(data_api.router)
app.include_router(evaluation_api.router)

@app.get("/health")
def health():
    return {"status":"ok"}
