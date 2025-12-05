# app/routers/evaluation_api.py
from fastapi import APIRouter, HTTPException
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
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

@router.get("/cluster/pca")
def get_cluster_pca_data():
    """Get PCA scatter plot data for clustering visualization (sampled for performance)."""
    try:
        pca_path = MODELS_DIR / "cluster_pca_data.joblib"
        if not pca_path.exists():
            raise HTTPException(404, "PCA data not found")
        
        pca_data = joblib.load(pca_path)
        X_pca = pca_data["X_pca"]
        labels = pca_data["labels"]
        evr = pca_data["explained_variance_ratio"]
        
        # Sample data for frontend performance (max 2000 points)
        n_samples = len(labels)
        max_samples = 2000
        if n_samples > max_samples:
            # Stratified sampling to keep cluster proportions
            np.random.seed(42)
            indices = []
            unique_labels = np.unique(labels)
            samples_per_cluster = max_samples // len(unique_labels)
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                n_take = min(samples_per_cluster, len(label_indices))
                indices.extend(np.random.choice(label_indices, n_take, replace=False))
            indices = np.array(indices)
        else:
            indices = np.arange(n_samples)
        
        # Build scatter data
        scatter_data = [
            {"x": float(X_pca[i, 0]), "y": float(X_pca[i, 1]), "cluster": int(labels[i])}
            for i in indices
        ]
        
        return {
            "scatter_data": scatter_data,
            "explained_variance": [float(evr[0]), float(evr[1])],
            "total_variance_explained": float(sum(evr)),
            "n_total": n_samples,
            "n_sampled": len(indices)
        }
    except Exception as e:
        raise HTTPException(500, f"Error loading PCA data: {str(e)}")


@router.get("/cluster/elbow")
def get_cluster_elbow_data():
    """Get elbow method data showing inertia and silhouette scores for various k values."""
    try:
        elbow_path = MODELS_DIR / "cluster_elbow_data.joblib"
        meta_path = MODELS_DIR / "meta_cluster.json"
        
        if not elbow_path.exists():
            raise HTTPException(404, "Elbow data not found. Please retrain the cluster model.")
        
        elbow_data = joblib.load(elbow_path)
        
        # Get the selected k value from metadata
        selected_k = 9  # default
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            selected_k = meta.get("n_clusters", 9)
        
        return {
            "elbow_data": elbow_data,
            "selected_k": selected_k,
            "description": "Inertia (WCSS) decreases as k increases. The 'elbow' point suggests optimal k."
        }
    except Exception as e:
        raise HTTPException(500, f"Error loading elbow data: {str(e)}")


@router.get("/croprec/per_class_metrics")
def get_croprec_per_class_metrics():
    """Get per-class precision, recall, F1 for crop recommendation."""
    try:
        meta_path = MODELS_DIR / "meta_croprec.json"
        if not meta_path.exists():
            raise HTTPException(404, "Crop recommendation metadata not found")
        
        meta = json.loads(meta_path.read_text())
        confusion = np.array(meta.get("confusion_matrix", []))
        classes = meta.get("classes", [])
        
        if confusion.size == 0 or len(classes) == 0:
            raise HTTPException(404, "Confusion matrix or classes not found")
        
        # Calculate per-class metrics from confusion matrix
        per_class_metrics = []
        for i, cls in enumerate(classes):
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = int(confusion[i, :].sum())
            
            per_class_metrics.append({
                "class": cls,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support
            })
        
        # Sort by F1 score (worst performing first for visibility)
        per_class_metrics.sort(key=lambda x: x["f1"])
        
        return {
            "per_class_metrics": per_class_metrics,
            "n_classes": len(classes)
        }
    except Exception as e:
        raise HTTPException(500, f"Error calculating per-class metrics: {str(e)}")


@router.get("/yield/predictions")
def get_yield_predictions():
    """Get actual vs predicted values for yield model visualization."""
    try:
        # Load model and data
        model = load_artifact("GradientBoosting_yield_model.joblib")
        df = pd.read_csv(DATA_PATH)
        
        NUM = ["N", "P", "K", "pH", "rainfall", "temperature"]
        CAT = ["State_Name", "Season", "Crop"]
        TARGET = "Yield_ton_per_hec"
        
        X = df[NUM + CAT].copy()
        y = df[TARGET].astype(float)
        
        # Use same split as training
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Sample for frontend (max 1500 points)
        n_samples = len(y_test)
        max_samples = 1500
        if n_samples > max_samples:
            np.random.seed(42)
            indices = np.random.choice(n_samples, max_samples, replace=False)
        else:
            indices = np.arange(n_samples)
        
        y_test_arr = y_test.values
        
        # Build scatter data for actual vs predicted
        scatter_data = [
            {"actual": float(y_test_arr[i]), "predicted": float(y_pred[i])}
            for i in indices
        ]
        
        # Build residual data
        residuals = y_pred - y_test_arr
        residual_data = [
            {"predicted": float(y_pred[i]), "residual": float(residuals[i])}
            for i in indices
        ]
        
        # Calculate residual distribution for histogram
        residual_hist, bin_edges = np.histogram(residuals, bins=30)
        residual_distribution = [
            {"bin": float((bin_edges[i] + bin_edges[i+1]) / 2), "count": int(residual_hist[i])}
            for i in range(len(residual_hist))
        ]
        
        return {
            "scatter_data": scatter_data,
            "residual_data": residual_data,
            "residual_distribution": residual_distribution,
            "n_total": n_samples,
            "n_sampled": len(indices),
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals))
        }
    except Exception as e:
        raise HTTPException(500, f"Error generating predictions: {str(e)}")
