"""K-Means clustering model."""

from typing import Dict, List, Any
import numpy as np
from sklearn.cluster import KMeans

from models.base import BaseClusterModel


class KMeansModel(BaseClusterModel):
    """
    K-Means clustering algorithm.

    Partitions data into k clusters by minimizing within-cluster variance.
    Good for spherical, evenly-sized clusters.
    """

    name = "K-Means"
    description = "Partition-based clustering that minimizes within-cluster variance"
    supports_n_clusters = True

    def __init__(self):
        super().__init__()
        self.params = {
            'n_clusters': 3,
            'init': 'k-means++',
            'n_init': 10,
            'max_iter': 300,
            'random_state': 42
        }

    def get_param_config(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'n_clusters',
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 20,
                'description': 'Number of clusters (k)'
            },
            {
                'name': 'init',
                'type': 'select',
                'default': 'k-means++',
                'options': ['k-means++', 'random'],
                'description': 'Initialization method'
            },
            {
                'name': 'n_init',
                'type': 'int',
                'default': 10,
                'min': 1,
                'max': 50,
                'description': 'Number of initializations'
            },
            {
                'name': 'max_iter',
                'type': 'int',
                'default': 300,
                'min': 100,
                'max': 1000,
                'description': 'Maximum iterations'
            }
        ]

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model = KMeans(
            n_clusters=self.params['n_clusters'],
            init=self.params['init'],
            n_init=self.params['n_init'],
            max_iter=self.params['max_iter'],
            random_state=self.params['random_state']
        )
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    def get_inertia(self) -> float:
        """Get within-cluster sum of squares (inertia)."""
        if self.model is not None:
            return self.model.inertia_
        return 0.0

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centroids."""
        if self.model is not None:
            return self.model.cluster_centers_
        return np.array([])
