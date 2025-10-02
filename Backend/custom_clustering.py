import numpy as np
from sklearn.metrics import silhouette_score

class KMeansScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def initialize_centroids(self, X):
        """k-means++ initialization"""
        np.random.seed(self.random_state)
        centroids = []
        centroids.append(X[np.random.randint(X.shape[0])])

        for _ in range(1, self.k):
            dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dist_sq / np.sum(dist_sq)
            cumulative = np.cumsum(probs)
            r = np.random.rand()
            for j, p in enumerate(cumulative):
                if r < p:
                    centroids.append(X[j])
                    break
        return np.array(centroids)

    def fit(self, X):
        """Train the KMeans model"""
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iters):
            labels = self.predict(X)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        """Assign clusters to points"""
        distances = np.array([np.linalg.norm(X - c, axis=1) for c in self.centroids])
        return np.argmin(distances, axis=0)

    def inertia(self, X):
        """Total squared distance from points to their cluster center"""
        return np.sum((X - self.centroids[self.labels_]) ** 2)

    def silhouette(self, X):
        """Silhouette score for quality of clustering"""
        return silhouette_score(X, self.labels_)
