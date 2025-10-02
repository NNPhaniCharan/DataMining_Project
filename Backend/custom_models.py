import numpy as np
from scipy.spatial.distance import cdist

class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # Avoid division by zero
        return self
    
    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RidgeRegression:
    """
    Ridge Regression with closed-form solution and optional feature engineering.
    Solves: w = (X^T X + alpha*I)^{-1} X^T y
    """
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=True, poly_degree=1):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.poly_degree = poly_degree
        self.coef_ = None
        self.intercept_ = None
        self.scaler_ = None
        
    def _create_polynomial_features(self, X):
        """Create polynomial features up to specified degree."""
        if self.poly_degree == 1:
            return X
        
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Generate all polynomial combinations
        poly_features = [X]
        for degree in range(2, self.poly_degree + 1):
            # Add interaction terms and powers
            for i in range(n_features):
                for combo in range(len(poly_features[degree-2].T)):
                    new_feature = X[:, i] * poly_features[degree-2][:, combo]
                    poly_features.append(new_feature.reshape(-1, 1))
        
        # Simple approach: just add squared terms for degree 2
        if self.poly_degree == 2:
            poly_features = [X, X**2]
        
        return np.hstack(poly_features)
        
    def fit(self, X, y):
        """
        Fit Ridge regression using closed-form solution.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        
        # Create polynomial features
        X = self._create_polynomial_features(X)
        
        # Normalize features
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        # Add intercept column if needed
        if self.fit_intercept:
            n_samples = X.shape[0]
            X = np.hstack([np.ones((n_samples, 1)), X])
        
        # Closed-form solution: w = (X^T X + alpha*I)^{-1} X^T y
        n_features = X.shape[1]
        reg_matrix = np.eye(n_features) * self.alpha
        if self.fit_intercept:
            reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Solve using more stable method
        try:
            weights = np.linalg.solve(XtX + reg_matrix, Xty)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for ill-conditioned matrices
            weights = np.linalg.lstsq(XtX + reg_matrix, Xty, rcond=None)[0]
        
        if self.fit_intercept:
            self.intercept_ = float(weights[0, 0])
            self.coef_ = weights[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.flatten()
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        y_pred : array, shape (n_samples,)
        """
        X = np.asarray(X)
        X = self._create_polynomial_features(X)
        
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return X @ self.coef_ + self.intercept_


class RidgeRegressionCV:
    """
    Ridge Regression with cross-validation for alpha selection.
    Uses shuffled k-fold CV for better generalization.
    """
    def __init__(self, alphas=None, cv=5, poly_degree=1, normalize=True):
        if alphas is None:
            alphas = np.logspace(-3, 3, 30)
        self.alphas = alphas
        self.cv = cv
        self.poly_degree = poly_degree
        self.normalize = normalize
        self.alpha_ = None
        self.best_model_ = None
        self.cv_scores_ = None
        
    def fit(self, X, y):
        """
        Fit Ridge regression with CV to select best alpha.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples = len(X)
        
        # Shuffle data for better CV
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        fold_size = n_samples // self.cv
        best_score = -np.inf
        best_alpha = self.alphas[0]
        all_scores = []
        
        # K-fold CV with shuffling
        for alpha in self.alphas:
            scores = []
            for fold in range(self.cv):
                # Create train/val split
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.cv - 1 else n_samples
                
                val_idx = np.arange(val_start, val_end)
                train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n_samples)])
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train and evaluate
                model = RidgeRegression(alpha=alpha, poly_degree=self.poly_degree, 
                                       normalize=self.normalize)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # R² score (handle edge cases)
                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-10))
                scores.append(r2)
            
            avg_score = np.mean(scores)
            all_scores.append(avg_score)
            
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha
        
        # Store CV results
        self.cv_scores_ = dict(zip(self.alphas, all_scores))
        
        # Train final model with best alpha on full data
        self.alpha_ = best_alpha
        self.best_model_ = RidgeRegression(alpha=best_alpha, poly_degree=self.poly_degree,
                                          normalize=self.normalize)
        self.best_model_.fit(X, y)
        
        return self
    
    def predict(self, X):
        """Predict using the best model."""
        return self.best_model_.predict(X)

    """
    K-Means Clustering with k-means++ initialization and feature scaling.
    """
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=1e-4,
                 random_state=None, normalize=True):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.normalize = normalize
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.scaler_ = None
        
    def _kmeans_plusplus_init(self, X, random_state):
        """
        Initialize cluster centers using k-means++ algorithm.
        """
        n_samples = X.shape[0]
        centers = [X[random_state.randint(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            # Compute distances from each point to nearest existing center
            distances = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            
            # Choose next center with probability proportional to distance²
            probabilities = distances / (distances.sum() + 1e-10)
            cumulative_probs = np.cumsum(probabilities)
            r = random_state.rand()
            
            for idx, cum_prob in enumerate(cumulative_probs):
                if r < cum_prob:
                    centers.append(X[idx])
                    break
        
        return np.array(centers)
    
    def _assign_clusters(self, X, centers):
        """Assign each point to nearest cluster center."""
        distances = cdist(X, centers, metric='euclidean')
        return np.argmin(distances, axis=1)
    
    def _update_centers(self, X, labels):
        """Update cluster centers as mean of assigned points."""
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centers[k] = cluster_points.mean(axis=0)
            else:
                # Reinitialize empty cluster at furthest point
                distances = np.min([np.sum((X - centers[j]) ** 2, axis=1) 
                                   for j in range(self.n_clusters) if j != k], axis=0)
                centers[k] = X[np.argmax(distances)]
        return centers
    
    def _compute_inertia(self, X, labels, centers):
        """Compute within-cluster sum of squares."""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centers[k]) ** 2)
        return inertia
    
    def fit(self, X, y=None):
        """
        Fit K-Means clustering.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : Ignored
        """
        X = np.asarray(X)
        
        # Normalize features (critical for K-Means!)
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random_state = np.random.RandomState(self.random_state)
        else:
            random_state = np.random.RandomState()
        
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        
        # Run k-means multiple times with different initializations
        for init_run in range(self.n_init):
            centers = self._kmeans_plusplus_init(X, random_state)
            
            # Run k-means iterations
            for iteration in range(self.max_iter):
                old_centers = centers.copy()
                labels = self._assign_clusters(X, centers)
                centers = self._update_centers(X, labels)
                
                # Check convergence based on center movement
                center_shift = np.sum((centers - old_centers) ** 2)
                if center_shift < self.tol:
                    break
            
            # Compute final inertia
            inertia = self._compute_inertia(X, labels, centers)
            
            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
        
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        labels : array, shape (n_samples,)
        """
        X = np.asarray(X)
        
        # Apply same normalization as training
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return self._assign_clusters(X, self.cluster_centers_)