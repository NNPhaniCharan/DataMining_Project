# app/routers/cluster_api.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from ..schemas import ClusterAssignRequest, ClusterAssignResponse, ClusterAssignItem, ClusterAssignFeatures
from ..deps import load_artifact

router = APIRouter(prefix="/cluster", tags=["cluster"])

pipe = None  # scaler + kmeans
cluster_profile = None  # optional: per-cluster means
pca = None  # PCA model
pca_data = None  # PCA transformed data

@router.on_event("startup")
def _load():
    global pipe, cluster_profile, pca, pca_data
    pipe = load_artifact("cluster_model.joblib")
    # try to load cluster profile (per-cluster means) if baked-in
    try:
        cluster_profile = load_artifact("cluster_profile.joblib")
    except Exception:
        cluster_profile = None
    # try to load PCA model and data
    try:
        pca = load_artifact("cluster_pca.joblib")
        pca_data = load_artifact("cluster_pca_data.joblib")
    except Exception:
        pca = None
        pca_data = None

@router.post("/assign", response_model=ClusterAssignResponse)
def assign_cluster(req: ClusterAssignRequest):
    try:
        df = pd.DataFrame([s.dict() for s in req.samples])[["N","P","K","pH","rainfall","temperature"]]
        cluster_ids = pipe.predict(df).tolist()
        return ClusterAssignResponse(assignments=[ClusterAssignItem(cluster_id=int(c)) for c in cluster_ids])
    except Exception as e:
        raise HTTPException(400, f"Clustering error: {e}")

@router.post("/visualization/pca")
def get_pca_visualization(user_sample: Optional[ClusterAssignFeatures] = None):
    """Generate and return PCA visualization of clusters as base64 image with optional user sample"""
    if pca_data is None or pca is None:
        raise HTTPException(404, "PCA data not available")
    
    try:
        X_pca = pca_data["X_pca"]
        labels = pca_data["labels"]
        explained_variance = pca_data["explained_variance_ratio"]
        
        # Transform user sample if provided
        user_pca = None
        user_cluster = None
        if user_sample:
            df = pd.DataFrame([user_sample.dict()])[["N","P","K","pH","rainfall","temperature"]]
            # Scale the sample
            scaler = pipe.named_steps["scaler"]
            user_scaled = scaler.transform(df)
            # Transform to PCA space
            user_pca = pca.transform(user_scaled)[0]
            # Get cluster assignment
            user_cluster = pipe.predict(df)[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique clusters
        n_clusters = len(np.unique(labels))
        
        # Use a colormap
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        # Plot each cluster with reduced alpha if user sample is provided
        alpha_value = 0.3 if user_pca is not None else 0.6
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            ax.scatter(
                X_pca[mask, 0], 
                X_pca[mask, 1],
                c=[colors[cluster_id]], 
                alpha=alpha_value,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )
        
        # Plot user sample if provided
        if user_pca is not None:
            # Plot the user's point with a distinct marker
            ax.scatter(
                user_pca[0], 
                user_pca[1],
                c='red',
                marker='*',
                s=800,
                edgecolors='darkred',
                linewidth=3,
                zorder=12
            )
            
            # Add annotation for user point
            ax.annotate(
                'YOUR\nINPUT',
                xy=user_pca,
                xytext=(20, 20),
                textcoords='offset points',
                ha='left',
                fontsize=11,
                fontweight='bold',
                color='darkred',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='darkred', linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkred', lw=2),
                zorder=13
            )
        
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)', fontsize=12, fontweight='bold')
        
        title_text = f'K-Means Clustering - PCA Visualization\n'
        title_text += f'Total Variance Explained: {explained_variance.sum():.1%}'
        if user_pca is not None:
            title_text += f' | Your Input → Cluster {user_cluster}'
        
        ax.set_title(
            title_text,
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Tight layout
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        response_data = {
            "image": f"data:image/png;base64,{img_base64}",
            "variance_explained": {
                "pc1": float(explained_variance[0]),
                "pc2": float(explained_variance[1]),
                "total": float(explained_variance.sum())
            }
        }
        
        if user_pca is not None:
            response_data["user_point"] = {
                "pca_coords": [float(user_pca[0]), float(user_pca[1])],
                "cluster_id": int(user_cluster)
            }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(500, f"Error generating PCA visualization: {str(e)}")