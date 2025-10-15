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


class DecisionTreeRegressor:
    """
    Simple Decision Tree Regressor for use in Gradient Boosting.
    Uses greedy algorithm to find best splits based on MSE reduction.
    """
    def __init__(self, max_depth=3, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        
    def _mse(self, y):
        """Calculate mean squared error for a set of values."""
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)
    
    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        best_mse_reduction = 0
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None
        
        current_mse = self._mse(y)
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Try a subset of thresholds for efficiency
            if len(thresholds) > 20:
                thresholds = np.percentile(feature_values, np.linspace(10, 90, 10))
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Check minimum samples per leaf
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate MSE reduction
                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                mse_reduction = current_mse - (left_mse + right_mse)
                
                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples = len(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return {'value': np.mean(y)}
        
        # Find best split
        feature_idx, threshold = self._find_best_split(X, y)
        
        if feature_idx is None:
            return {'value': np.mean(y)}
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """Build decision tree."""
        X = np.asarray(X)
        y = np.asarray(y)
        self.tree_ = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, tree):
        """Predict a single sample by traversing the tree."""
        if 'value' in tree:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Predict for all samples."""
        X = np.asarray(X)
        return np.array([self._predict_sample(x, self.tree_) for x in X])


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor implemented from scratch.
    
    Uses sequential ensemble of decision trees where each tree fits
    the residuals (negative gradients) of the previous predictions.
    
    Algorithm:
    1. Initialize with mean of target
    2. For each boosting iteration:
       a. Compute residuals (negative gradients)
       b. Fit a decision tree to residuals
       c. Update predictions with learning_rate * tree_prediction
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=10, min_samples_leaf=5, subsample=1.0,
                 random_state=None, verbose=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample  # Stochastic GB: fraction of samples per tree
        self.random_state = random_state
        self.verbose = verbose
        
        self.init_prediction_ = None
        self.trees_ = []
        self.train_score_ = []
        self.feature_importances_ = None
        
    def fit(self, X, y):
        """
        Fit gradient boosting model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = len(X)
        
        # Initialize with mean prediction
        self.init_prediction_ = np.mean(y)
        predictions = np.full(n_samples, self.init_prediction_)
        
        # Build trees sequentially
        for i in range(self.n_estimators):
            # Compute negative gradients (residuals for MSE loss)
            residuals = y - predictions
            
            # Subsample for stochastic gradient boosting
            if self.subsample < 1.0:
                sample_indices = np.random.choice(
                    n_samples, 
                    size=int(n_samples * self.subsample),
                    replace=False
                )
                X_sample = X[sample_indices]
                residuals_sample = residuals[sample_indices]
            else:
                X_sample = X
                residuals_sample = residuals
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_sample, residuals_sample)
            
            # Update predictions with learning rate
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions
            
            # Store tree
            self.trees_.append(tree)
            
            # Track training performance
            mse = np.mean((y - predictions) ** 2)
            self.train_score_.append(mse)
            
            if self.verbose > 0 and (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{self.n_estimators}, MSE: {mse:.4f}")
        
        # Compute feature importances (average across all trees)
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)
        
        for tree in self.trees_:
            tree_importances = self._get_tree_feature_importances(tree.tree_, n_features)
            self.feature_importances_ += tree_importances
        
        # Normalize to sum to 1
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        return self
    
    def _get_tree_feature_importances(self, node, n_features, total_samples=None):
        """Recursively compute feature importances for a tree."""
        importances = np.zeros(n_features)
        
        if node is None or 'value' in node:
            return importances
        
        # Node has a split
        if 'feature' in node:
            # Contribution of this split (based on number of samples if available)
            importances[node['feature']] += 1.0
            
            # Recurse to children
            if 'left' in node:
                importances += self._get_tree_feature_importances(node['left'], n_features, total_samples)
            if 'right' in node:
                importances += self._get_tree_feature_importances(node['right'], n_features, total_samples)
        
        return importances
    
    def predict(self, X):
        """
        Predict using the gradient boosting model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        y_pred : array, shape (n_samples,)
        """
        X = np.asarray(X)
        
        # Start with initial prediction
        predictions = np.full(len(X), self.init_prediction_)
        
        # Add contribution from each tree
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
    
    def staged_predict(self, X):
        """
        Predict at each stage (useful for early stopping evaluation).
        
        Yields predictions after each tree is added.
        """
        X = np.asarray(X)
        predictions = np.full(len(X), self.init_prediction_)
        
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)
            yield predictions.copy()


class GradientBoostingRegressorCV:
    """
    Gradient Boosting with cross-validation for hyperparameter tuning.
    Uses early stopping on validation set to determine optimal n_estimators.
    """
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3,
                 min_samples_split=10, min_samples_leaf=5, subsample=0.8,
                 cv=5, random_state=None, verbose=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        
        self.best_n_estimators_ = None
        self.best_model_ = None
        self.cv_scores_ = []
        
    def fit(self, X, y):
        """
        Fit with cross-validation to find optimal n_estimators.
        
        Uses early stopping: monitors validation MSE and stops when
        it doesn't improve for multiple iterations.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = len(X)
        fold_size = n_samples // self.cv
        
        # Shuffle data
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        cv_scores_per_iter = []
        
        # K-fold CV
        for fold in range(self.cv):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.cv - 1 else n_samples
            
            val_idx = np.arange(val_start, val_end)
            train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n_samples)])
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                subsample=self.subsample,
                random_state=self.random_state,
                verbose=0
            )
            model.fit(X_train, y_train)
            
            # Evaluate at each stage
            fold_scores = []
            for pred in model.staged_predict(X_val):
                mse = np.mean((y_val - pred) ** 2)
                fold_scores.append(mse)
            
            cv_scores_per_iter.append(fold_scores)
        
        # Average scores across folds
        avg_scores = np.mean(cv_scores_per_iter, axis=0)
        self.cv_scores_ = avg_scores
        
        # Find best number of estimators (lowest validation MSE)
        self.best_n_estimators_ = int(np.argmin(avg_scores)) + 1
        
        if self.verbose > 0:
            print(f"Best n_estimators from CV: {self.best_n_estimators_}")
            print(f"Best CV MSE: {avg_scores[self.best_n_estimators_-1]:.4f}")
        
        # Train final model on full data with best n_estimators
        self.best_model_ = GradientBoostingRegressor(
            n_estimators=self.best_n_estimators_,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.best_model_.fit(X, y)
        
        return self
    
    def predict(self, X):
        """Predict using the best model."""
        return self.best_model_.predict(X)
    
    @property
    def feature_importances_(self):
        """Get feature importances from the best model."""
        if self.best_model_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.best_model_.feature_importances_


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


class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', metric='minkowski', 
                 p=2, normalize=True):
        self.n_neighbors = n_neighbors
        self.weights = weights  # 'uniform' or 'distance'
        self.metric = metric
        self.p = p
        self.normalize = normalize
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self.scaler_ = None
        
    def fit(self, X, y):
        X = np.asarray(X)
        
        # Normalize features (critical for KNN!)
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        self.X_train_ = X
        self.y_train_ = np.asarray(y)
        self.classes_, y_encoded = np.unique(self.y_train_, return_inverse=True)
        self._y = y_encoded
        
        return self
    
    def _compute_distances(self, X):
        """Compute distances between X and training data."""
        if self.metric == 'minkowski':
            if self.p == 2:
                return cdist(X, self.X_train_, metric='euclidean')
            else:
                return cdist(X, self.X_train_, metric='minkowski', p=self.p)
        else:
            return cdist(X, self.X_train_, metric=self.metric)
    
    def predict(self, X):
        X = np.asarray(X)
        
        # Apply same normalization as training
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        distances = self._compute_distances(X)
        
        predictions = []
        for i in range(len(X)):
            # Get k nearest neighbors
            nearest_indices = np.argpartition(distances[i], self.n_neighbors)[:self.n_neighbors]
            nearest_indices = nearest_indices[np.argsort(distances[i][nearest_indices])]
            nearest_labels = self.y_train_[nearest_indices]
            
            if self.weights == 'uniform':
                # Simple majority vote
                unique, counts = np.unique(nearest_labels, return_counts=True)
                predictions.append(unique[np.argmax(counts)])
            elif self.weights == 'distance':
                # Distance-weighted vote
                nearest_distances = distances[i][nearest_indices]
                # Avoid division by zero with small epsilon
                weights = 1 / (nearest_distances + 1e-10)
                
                # Weighted vote for each class
                class_weights = {}
                for label, weight in zip(nearest_labels, weights):
                    class_weights[label] = class_weights.get(label, 0) + weight
                
                predictions.append(max(class_weights.items(), key=lambda x: x[1])[0])
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        X = np.asarray(X)
        
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        distances = self._compute_distances(X)
        n_classes = len(self.classes_)
        probas = np.zeros((len(X), n_classes))
        
        for i in range(len(X)):
            nearest_indices = np.argpartition(distances[i], self.n_neighbors)[:self.n_neighbors]
            nearest_indices = nearest_indices[np.argsort(distances[i][nearest_indices])]
            nearest_labels = self.y_train_[nearest_indices]
            
            if self.weights == 'uniform':
                for label in nearest_labels:
                    class_idx = np.where(self.classes_ == label)[0][0]
                    probas[i, class_idx] += 1.0 / self.n_neighbors
            else:
                nearest_distances = distances[i][nearest_indices]
                weights = 1 / (nearest_distances + 1e-10)
                weight_sum = weights.sum()
                
                for label, weight in zip(nearest_labels, weights):
                    class_idx = np.where(self.classes_ == label)[0][0]
                    probas[i, class_idx] += weight / weight_sum
        
        return probas

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Return distances and indices of nearest neighbors."""
        X = np.asarray(X)
        
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
            
        distances = self._compute_distances(X)
        indices = np.argsort(distances, axis=1)[:, :n_neighbors]
        
        if return_distance:
            row_idx = np.arange(X.shape[0])[:, None]
            dists = distances[row_idx, indices]
            return dists, indices
        return indices


class KMeansClustering:
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
