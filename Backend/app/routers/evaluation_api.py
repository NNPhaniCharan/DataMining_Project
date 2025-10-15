# app/routers/evaluation_api.py
from fastapi import APIRouter, HTTPException
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from ..deps import BASE_DIR, load_artifact, MODELS_DIR

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

DATA_PATH = BASE_DIR / "DataAndCleaning" / "Data" / "CleanedData" / "Crop_production_cleaned.csv"

@router.get("/yield/metrics")
def get_yield_metrics():
    """Get yield prediction model metrics and coefficients."""
    try:
        meta_path = MODELS_DIR / "meta_GradientBoosting_yield.json"
        if not meta_path.exists():
            raise HTTPException(404, "Yield model metadata not found")
        
        meta = json.loads(meta_path.read_text())
        
        # Load model to extract feature importances
        model = load_artifact("GradientBoosting_yield_model.joblib")
        gb_model = model.named_steps["model"]
        
        # Get feature names from preprocessor
        preprocessor = model.named_steps["pre"]
        num_features = preprocessor.transformers_[0][2]  # numerical features
        cat_transformer = preprocessor.transformers_[1][1]  # categorical transformer
        cat_features = preprocessor.transformers_[1][2]
        
        # Get categorical feature names after one-hot encoding
        try:
            cat_feature_names = []
            for i, cat_col in enumerate(cat_features):
                categories = cat_transformer.categories_[i]
                cat_feature_names.extend([f"{cat_col}_{cat}" for cat in categories])
        except:
            cat_feature_names = [f"cat_{i}" for i in range(len(gb_model.feature_importances_) - len(num_features))]
        
        all_feature_names = list(num_features) + cat_feature_names
        
        # Create feature importance data (Gradient Boosting uses importances, not coefficients)
        feature_importances = [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(all_feature_names, gb_model.feature_importances_)
        ]
        
        # Sort by importance (descending)
        feature_importances.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "metrics": {
                "r2": meta.get("r2"),
                "rmse": meta.get("rmse"),
                "mae": meta.get("mae"),
                "best_n_estimators": meta.get("best_n_estimators"),
                "learning_rate": meta.get("learning_rate"),
                "max_depth": meta.get("max_depth"),
                "n_train": meta.get("n_train"),
                "n_test": meta.get("n_test"),
                "algorithm": meta.get("algorithm", "Custom Gradient Boosting")
            },
            "feature_importances": feature_importances[:50],  # Top 50 features
            "total_features": len(all_feature_names)
        }
    except Exception as e:
        raise HTTPException(500, f"Error loading yield metrics: {str(e)}")

@router.get("/croprec/metrics")
def get_croprec_metrics():
    """Get crop recommendation model metrics and confusion matrix."""
    try:
        meta_path = MODELS_DIR / "meta_croprec.json"
        if not meta_path.exists():
            raise HTTPException(404, "Crop recommendation model metadata not found")
        
        meta = json.loads(meta_path.read_text())
        
        # Load model
        model = load_artifact("croprec_model.joblib")
        knn_classifier = model.named_steps["clf"]
        
        return {
            "metrics": {
                "top1_accuracy": meta.get("top1_accuracy"),
                "top3_accuracy": meta.get("top3_accuracy"),
                "f1_macro": meta.get("f1_macro"),
                "n_train": meta.get("n_train"),
                "n_test": meta.get("n_test"),
                "n_neighbors": knn_classifier.n_neighbors,
                "algorithm": meta.get("algorithm", "Custom k-NN Classifier")
            },
            "confusion_matrix": meta.get("confusion_matrix"),
            "classes": knn_classifier.classes_.tolist() if hasattr(knn_classifier, 'classes_') else [],
            "n_classes": len(knn_classifier.classes_) if hasattr(knn_classifier, 'classes_') else 0
        }
    except Exception as e:
        raise HTTPException(500, f"Error loading crop rec metrics: {str(e)}")

@router.get("/cluster/metrics")
def get_cluster_metrics():
    """Get clustering model metrics and cluster centers."""
    try:
        meta_path = MODELS_DIR / "meta_cluster.json"
        profile_path = MODELS_DIR / "cluster_profile.joblib"
        
        if not meta_path.exists():
            raise HTTPException(404, "Cluster model metadata not found")
        
        meta = json.loads(meta_path.read_text())
        
        # Load cluster profile if available
        cluster_centers = []
        if profile_path.exists():
            import joblib
            profile = joblib.load(profile_path)
            features = profile.get("features", [])
            centers_unscaled = profile.get("centers_unscaled", [])
            cluster_profiles_data = profile.get("cluster_profiles", [])
            
            for i, center in enumerate(centers_unscaled):
                cluster_data = {"cluster_id": i}
                for j, feat in enumerate(features):
                    cluster_data[feat] = float(center[j])
                cluster_data["count"] = meta.get("counts", [])[i] if i < len(meta.get("counts", [])) else 0
                
                # Add yield and crop profile if available
                if i < len(cluster_profiles_data):
                    profile_data = cluster_profiles_data[i]
                    cluster_data["avg_yield"] = profile_data.get("avg_yield", 0)
                    cluster_data["median_yield"] = profile_data.get("median_yield", 0)
                    cluster_data["min_yield"] = profile_data.get("min_yield", 0)
                    cluster_data["max_yield"] = profile_data.get("max_yield", 0)
                    cluster_data["top_crops"] = profile_data.get("top_crops", [])
                
                cluster_centers.append(cluster_data)
        
        return {
            "metrics": {
                "n_clusters": meta.get("n_clusters"),
                "counts": meta.get("counts", []),
                "silhouette_score": meta.get("silhouette_score"),
                "calinski_harabasz_score": meta.get("calinski_harabasz_score"),
                "algorithm": meta.get("algorithm", "Custom K-Means with k-means++")
            },
            "cluster_centers": cluster_centers,
            "features": profile.get("features", []) if profile_path.exists() else []
        }
    except Exception as e:
        raise HTTPException(500, f"Error loading cluster metrics: {str(e)}")

@router.get("/summary")
def get_evaluation_summary():
    """Get summary of all models."""
    try:
        yield_meta = json.loads((MODELS_DIR / "meta_GradientBoosting_yield.json").read_text()) if (MODELS_DIR / "meta_GradientBoosting_yield.json").exists() else {}
        crop_meta = json.loads((MODELS_DIR / "meta_croprec.json").read_text()) if (MODELS_DIR / "meta_croprec.json").exists() else {}
        cluster_meta = json.loads((MODELS_DIR / "meta_cluster.json").read_text()) if (MODELS_DIR / "meta_cluster.json").exists() else {}
        
        return {
            "yield_prediction": {
                "algorithm": yield_meta.get("algorithm", "Custom Gradient Boosting"),
                "r2": yield_meta.get("r2"),
                "rmse": yield_meta.get("rmse"),
                "mae": yield_meta.get("mae"),
                "best_n_estimators": yield_meta.get("best_n_estimators")
            },
            "crop_recommendation": {
                "algorithm": crop_meta.get("algorithm", "Custom k-NN Classifier"),
                "top1_accuracy": crop_meta.get("top1_accuracy"),
                "top3_accuracy": crop_meta.get("top3_accuracy"),
                "f1_macro": crop_meta.get("f1_macro"),
                "n_train": crop_meta.get("n_train")
            },
            "clustering": {
                "algorithm": cluster_meta.get("algorithm", "Custom K-Means"),
                "n_clusters": cluster_meta.get("n_clusters"),
                "silhouette_score": cluster_meta.get("silhouette_score"),
                "calinski_harabasz_score": cluster_meta.get("calinski_harabasz_score"),
                "counts": cluster_meta.get("counts", [])
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Error loading evaluation summary: {str(e)}")
