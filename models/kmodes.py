"""
K-Modes Clustering with Jaccard Distance.

K-Modes is the categorical analogue of K-Means, using modes instead of means
and Jaccard/Hamming distance instead of Euclidean distance.

For poverty/deprivation data, Jaccard distance is preferred because:
- Two households both LACKING a deprivation (0-0 match) is less informative
- Two households SHARING a deprivation (1-1 match) is more informative
- Jaccard handles this asymmetry; Euclidean does not
"""

import numpy as np
from typing import Dict, List, Any, Optional
from models.base import BaseClusterModel


class KModesModel(BaseClusterModel):
    """
    K-Modes clustering for categorical/binary data.

    Uses frequency-based modes instead of means, with Jaccard or
    Hamming distance for similarity measurement.
    """

    name = "K-Modes (Categorical)"
    description = "Mode-based clustering for binary/categorical data with Jaccard distance"
    supports_n_clusters = True

    def __init__(self):
        super().__init__()
        self.params = {
            'n_clusters': 3,
            'max_iter': 100,
            'n_init': 10,
            'distance_metric': 'jaccard',
            'init_method': 'huang',
            'random_state': 42
        }
        self.cluster_modes_ = None
        self.costs_ = []

    def get_param_config(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'n_clusters',
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 10,
                'description': 'Number of clusters'
            },
            {
                'name': 'max_iter',
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 500,
                'description': 'Maximum iterations'
            },
            {
                'name': 'n_init',
                'type': 'int',
                'default': 10,
                'min': 1,
                'max': 50,
                'description': 'Number of random initializations'
            },
            {
                'name': 'distance_metric',
                'type': 'select',
                'default': 'jaccard',
                'options': ['jaccard', 'dice', 'hamming'],
                'description': 'Distance metric for binary data'
            },
            {
                'name': 'init_method',
                'type': 'select',
                'default': 'huang',
                'options': ['huang', 'cao', 'random'],
                'description': 'Initialization method'
            }
        ]

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def _binarize(self, X: np.ndarray) -> np.ndarray:
        """Convert data to binary format (0/1)."""
        X_binary = np.zeros_like(X)
        for j in range(X.shape[1]):
            col = X[:, j]
            unique_vals = np.unique(col[~np.isnan(col)])
            if len(unique_vals) <= 2:
                X_binary[:, j] = (col > np.min(unique_vals)).astype(int)
            else:
                median_val = np.nanmedian(col)
                X_binary[:, j] = (col > median_val).astype(int)
        return X_binary.astype(int)

    def _jaccard_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Jaccard distance between two binary vectors.

        D_Jaccard(A, B) = 1 - |A ∩ B| / |A ∪ B|

        For poverty data: focuses on shared deprivations (1-1 matches)
        """
        intersection = np.sum((a == 1) & (b == 1))
        union = np.sum((a == 1) | (b == 1))

        if union == 0:
            return 0.0  # Both vectors are all zeros

        return 1.0 - (intersection / union)

    def _dice_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Dice distance (Sorensen-Dice coefficient).

        D_Dice(A, B) = 1 - 2|A ∩ B| / (|A| + |B|)
        """
        intersection = np.sum((a == 1) & (b == 1))
        sum_ones = np.sum(a == 1) + np.sum(b == 1)

        if sum_ones == 0:
            return 0.0

        return 1.0 - (2 * intersection / sum_ones)

    def _hamming_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate Hamming distance (proportion of mismatches).
        """
        return np.mean(a != b)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate distance based on selected metric."""
        metric = self.params['distance_metric']
        if metric == 'jaccard':
            return self._jaccard_distance(a, b)
        elif metric == 'dice':
            return self._dice_distance(a, b)
        else:  # hamming
            return self._hamming_distance(a, b)

    def _compute_mode(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the mode (most frequent value) for each feature.
        """
        n_features = X.shape[1]
        mode = np.zeros(n_features, dtype=int)

        for j in range(n_features):
            values, counts = np.unique(X[:, j], return_counts=True)
            mode[j] = values[np.argmax(counts)]

        return mode

    def _init_huang(self, X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
        """
        Huang initialization: initialize modes based on attribute frequency.
        """
        n_samples, n_features = X.shape
        modes = np.zeros((k, n_features), dtype=int)

        # Calculate attribute frequencies
        for j in range(n_features):
            values, counts = np.unique(X[:, j], return_counts=True)
            probs = counts / counts.sum()

            for c in range(k):
                modes[c, j] = rng.choice(values, p=probs)

        return modes

    def _init_cao(self, X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
        """
        Cao initialization: maximize initial mode diversity.
        """
        n_samples, n_features = X.shape

        # First mode: random sample
        first_idx = rng.randint(n_samples)
        modes = [X[first_idx].copy()]

        # Subsequent modes: maximize distance to existing modes
        for _ in range(k - 1):
            max_min_dist = -1
            best_idx = 0

            for i in range(n_samples):
                min_dist = min(self._distance(X[i], mode) for mode in modes)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            modes.append(X[best_idx].copy())

        return np.array(modes, dtype=int)

    def _init_random(self, X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
        """
        Random initialization: select k random samples as initial modes.
        """
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=k, replace=False)
        return X[indices].copy()

    def _initialize_modes(self, X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
        """Initialize cluster modes based on selected method."""
        method = self.params['init_method']

        if method == 'huang':
            return self._init_huang(X, k, rng)
        elif method == 'cao':
            return self._init_cao(X, k, rng)
        else:  # random
            return self._init_random(X, k, rng)

    def _assign_clusters(self, X: np.ndarray, modes: np.ndarray) -> np.ndarray:
        """Assign each sample to the nearest mode."""
        n_samples = X.shape[0]
        k = modes.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            min_dist = float('inf')
            for c in range(k):
                dist = self._distance(X[i], modes[c])
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = c

        return labels

    def _update_modes(self, X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        """Update modes based on current cluster assignments."""
        n_features = X.shape[1]
        modes = np.zeros((k, n_features), dtype=int)

        for c in range(k):
            cluster_members = X[labels == c]
            if len(cluster_members) > 0:
                modes[c] = self._compute_mode(cluster_members)
            else:
                # Empty cluster: reinitialize randomly
                modes[c] = X[np.random.randint(X.shape[0])]

        return modes

    def _compute_cost(self, X: np.ndarray, labels: np.ndarray, modes: np.ndarray) -> float:
        """Compute total clustering cost (sum of distances to modes)."""
        total_cost = 0.0
        for i, label in enumerate(labels):
            total_cost += self._distance(X[i], modes[label])
        return total_cost

    def _fit_single(self, X: np.ndarray, random_state: int) -> Dict:
        """Run K-Modes once with a given random state."""
        k = self.params['n_clusters']
        rng = np.random.RandomState(random_state)

        # Initialize modes
        modes = self._initialize_modes(X, k, rng)

        prev_cost = float('inf')
        costs = []

        for iteration in range(self.params['max_iter']):
            # Assign clusters
            labels = self._assign_clusters(X, modes)

            # Update modes
            modes = self._update_modes(X, labels, k)

            # Compute cost
            cost = self._compute_cost(X, labels, modes)
            costs.append(cost)

            # Check convergence
            if abs(prev_cost - cost) < 1e-6:
                break

            prev_cost = cost

        return {
            'labels': labels,
            'modes': modes,
            'cost': cost,
            'costs': costs,
            'n_iterations': iteration + 1
        }

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit K-Modes model and return cluster labels.

        Args:
            X: Feature array (will be binarized internally)

        Returns:
            Array of cluster labels (0 to k-1)
        """
        # Binarize data
        X_binary = self._binarize(X)

        # Run multiple initializations
        best_result = None
        best_cost = float('inf')

        base_random_state = self.params.get('random_state', 42)

        for init in range(self.params['n_init']):
            result = self._fit_single(X_binary, base_random_state + init)

            if result['cost'] < best_cost:
                best_cost = result['cost']
                best_result = result

        # Store results
        self.cluster_modes_ = best_result['modes']
        self.labels_ = best_result['labels']
        self.costs_ = best_result['costs']
        self.final_cost_ = best_result['cost']

        return self.labels_

    def get_cluster_modes(self) -> np.ndarray:
        """Get the cluster modes (categorical centroids)."""
        return self.cluster_modes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        X_binary = self._binarize(X)
        return self._assign_clusters(X_binary, self.cluster_modes_)


class HierarchicalJaccardModel(BaseClusterModel):
    """
    Hierarchical Agglomerative Clustering with Jaccard distance.

    Uses asymmetric binary distance metrics appropriate for
    deprivation/poverty data.
    """

    name = "Hierarchical (Jaccard)"
    description = "Agglomerative clustering with Jaccard distance for binary data"
    supports_n_clusters = True

    def __init__(self):
        super().__init__()
        self.params = {
            'n_clusters': 3,
            'linkage': 'average',
            'distance_metric': 'jaccard'
        }

    def get_param_config(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'n_clusters',
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 10,
                'description': 'Number of clusters'
            },
            {
                'name': 'linkage',
                'type': 'select',
                'default': 'average',
                'options': ['average', 'complete', 'single'],
                'description': 'Linkage criterion'
            },
            {
                'name': 'distance_metric',
                'type': 'select',
                'default': 'jaccard',
                'options': ['jaccard', 'dice', 'hamming'],
                'description': 'Distance metric'
            }
        ]

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def _binarize(self, X: np.ndarray) -> np.ndarray:
        """Convert data to binary format (0/1)."""
        X_binary = np.zeros_like(X)
        for j in range(X.shape[1]):
            col = X[:, j]
            unique_vals = np.unique(col[~np.isnan(col)])
            if len(unique_vals) <= 2:
                X_binary[:, j] = (col > np.min(unique_vals)).astype(int)
            else:
                median_val = np.nanmedian(col)
                X_binary[:, j] = (col > median_val).astype(int)
        return X_binary.astype(int)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit hierarchical clustering with Jaccard distance.

        Args:
            X: Feature array

        Returns:
            Array of cluster labels
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        # Binarize data
        X_binary = self._binarize(X)

        # Compute pairwise distances
        metric = self.params['distance_metric']
        if metric == 'jaccard':
            dist_matrix = pdist(X_binary, metric='jaccard')
        elif metric == 'dice':
            dist_matrix = pdist(X_binary, metric='dice')
        else:  # hamming
            dist_matrix = pdist(X_binary, metric='hamming')

        # Handle NaN/Inf in distance matrix
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=1.0, neginf=0.0)

        # Hierarchical clustering
        Z = linkage(dist_matrix, method=self.params['linkage'])

        # Cut dendrogram to get clusters
        self.labels_ = fcluster(Z, t=self.params['n_clusters'], criterion='maxclust') - 1

        return self.labels_
