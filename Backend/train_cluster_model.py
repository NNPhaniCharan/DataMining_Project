# train_cluster_model.py
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from custom_models import KMeansClustering

# Resolve paths relative to the repository root so the script works
# whether it's run from Backend/ or the project root.
BASE = Path(__file__).resolve().parent
# The cleaned CSV sits inside Backend/DataAndCleaning/Data/CleanedData
DATA = BASE / "DataAndCleaning" / "Data" / "CleanedData" / "Crop_production_cleaned.csv"
MODEL = BASE / "models" / "cluster_model.joblib"
PROFILE = BASE / "models" / "cluster_profile.joblib"
META = BASE / "models" / "meta_cluster.json"
PCA_DATA = BASE / "models" / "cluster_pca_data.joblib"
ELBOW_DATA = BASE / "models" / "cluster_elbow_data.joblib"

FEATS = ["N","P","K","pH","rainfall","temperature"]
N_CLUSTERS = 9
ELBOW_K_RANGE = range(2, 16)  # k values for elbow graph

def compute_elbow_data(X_scaled):
    """Compute inertia and silhouette scores for various k values."""
    elbow_data = []
    print("\nComputing elbow data for k = 2 to 15...")
    
    for k in ELBOW_K_RANGE:
        kmeans = KMeansClustering(n_clusters=k, n_init=5, random_state=42, normalize=False)
        kmeans.fit(X_scaled)
        
        # Compute silhouette score (skip for k=2 which is trivial)
        sil_score = float(silhouette_score(X_scaled, kmeans.labels_)) if k > 2 else 0
        
        elbow_data.append({
            "k": k,
            "inertia": float(kmeans.inertia_),
            "silhouette": sil_score
        })
        print(f"  k={k}: inertia={kmeans.inertia_:.2f}, silhouette={sil_score:.4f}")
    
    return elbow_data

def main():
    df = pd.read_csv(DATA)
    X = df[FEATS].copy()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeansClustering(n_clusters=N_CLUSTERS, n_init=10, random_state=42))
    ])
    pipe.fit(X)

    labels = pipe.named_steps["kmeans"].labels_
    centers = pipe.named_steps["kmeans"].cluster_centers_  # in scaled space
    
    # Get scaled features for evaluation metrics
    X_scaled = pipe.named_steps["scaler"].transform(X)
    
    # Calculate clustering evaluation metrics
    silhouette = float(silhouette_score(X_scaled, labels))
    calinski_harabasz = float(calinski_harabasz_score(X_scaled, labels))

    # Create human-friendly profile in original feature scale
    scaler = pipe.named_steps["scaler"]
    centers_unscaled = scaler.inverse_transform(centers)
    
    # Add cluster labels to the original dataframe
    df['cluster'] = labels
    
    # Calculate cluster profiles with yields and dominant crops
    cluster_profiles = []
    for cluster_id in range(N_CLUSTERS):
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Calculate yield statistics
        avg_yield = float(cluster_df['Yield_ton_per_hec'].mean())
        median_yield = float(cluster_df['Yield_ton_per_hec'].median())
        min_yield = float(cluster_df['Yield_ton_per_hec'].min())
        max_yield = float(cluster_df['Yield_ton_per_hec'].max())
        
        # Find top 5 dominant crops by count
        crop_counts = cluster_df['Crop'].value_counts()
        top_crops = [
            {
                "crop": crop,
                "count": int(count),
                "percentage": float(count / len(cluster_df) * 100)
            }
            for crop, count in crop_counts.head(5).items()
        ]
        
        # Calculate average yield per top crop
        for crop_info in top_crops:
            crop_name = crop_info["crop"]
            crop_yield = cluster_df[cluster_df['Crop'] == crop_name]['Yield_ton_per_hec'].mean()
            crop_info["avg_yield"] = float(crop_yield)
        
        cluster_profiles.append({
            "cluster_id": cluster_id,
            "avg_yield": avg_yield,
            "median_yield": median_yield,
            "min_yield": min_yield,
            "max_yield": max_yield,
            "top_crops": top_crops
        })
    
    profile = {
        "features": FEATS,
        "centers_unscaled": centers_unscaled.tolist(),
        "counts": np.bincount(labels).tolist(),
        "cluster_profiles": cluster_profiles
    }

    # Generate PCA data for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_data = {
        "X_pca": X_pca,
        "labels": labels,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
    }

    # Compute elbow data for k selection visualization
    elbow_data = compute_elbow_data(X_scaled)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL)
    joblib.dump(profile, PROFILE)
    joblib.dump(pca_data, PCA_DATA)
    joblib.dump(elbow_data, ELBOW_DATA)
    META.write_text(json.dumps({
        "n_clusters": N_CLUSTERS,
        "counts": profile["counts"],
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski_harabasz,
        "algorithm": "Custom K-Means with k-means++ initialization"
    }, indent=2))
    print("Saved:", MODEL, PROFILE, PCA_DATA, ELBOW_DATA)
    print(f"\nMetrics:")
    print(f"  PCA Variance Explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz Index: {calinski_harabasz:.2f}")
    print("\nCluster Profiles:")
    for cp in cluster_profiles:
        print(f"\nCluster {cp['cluster_id']}: Avg Yield={cp['avg_yield']:.2f} t/ha")
        print(f"  Top crops: {', '.join([c['crop'] for c in cp['top_crops'][:3]])}")

if __name__ == "__main__":
    main()
