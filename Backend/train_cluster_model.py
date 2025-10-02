import pandas as pd
import json
import matplotlib.pyplot as plt
from custom_clustering import KMeansScratch

# 1. Load cleaned dataset
data = pd.read_csv("Backend/DataAndCleaning/Data/CleanedData/Crop_production_cleaned.csv")

# 2. Expected features for clustering (include both uppercase and lowercase variations)
expected_features = ['N', 'P', 'K', 'pH', 'Rainfall', 'Temperature', 'rainfall', 'temperature']

# 3. Keep only the features that actually exist in the dataset
available_features = [col for col in expected_features if col in data.columns]

if not available_features:
    raise ValueError("❌ None of the expected features found in dataset!")

print("✅ Using features for clustering:", available_features)

X = data[available_features].values

# 4. Evaluate for multiple k values
inertias, silhouettes = [], []
k_values = range(2, 10)

for k in k_values:
    model = KMeansScratch(k=k).fit(X)
    inertias.append(model.inertia(X))
    silhouettes.append(model.silhouette(X))

# Save Elbow Plot
plt.plot(k_values, inertias, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.savefig("Backend/models/elbow_curve.png")
plt.close()

# Save Silhouette Plot
plt.plot(k_values, silhouettes, marker='o')
plt.title("Silhouette Scores for Different k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.savefig("Backend/models/silhouette_scores.png")
plt.close()

# 5. Pick the best k (max silhouette)
best_k = k_values[silhouettes.index(max(silhouettes))]
final_model = KMeansScratch(k=best_k).fit(X)

# 6. Save results to JSON
meta_clusters = {
    "used_features": available_features,
    "best_k": best_k,
    "centroids": final_model.centroids.tolist(),
    "silhouette_score": final_model.silhouette(X)
}

with open("Backend/models/meta_clusters.json", "w") as f:
    json.dump(meta_clusters, f, indent=4)

print("✅ Clustering completed.")
print(f"Best k = {best_k}, silhouette = {meta_clusters['silhouette_score']:.4f}")
print("Results saved in Backend/models/meta_clusters.json")
