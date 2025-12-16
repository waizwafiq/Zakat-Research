"""Spectral Clustering model."""

from typing import Dict, List, Any
import numpy as np
from sklearn.cluster import SpectralClustering

from models.base import BaseClusterModel


class SpectralModel(BaseClusterModel):
    """
    Spectral Clustering.

    Uses graph theory (eigenvalues of the Laplacian matrix) to perform dimensionality
    reduction before clustering in fewer dimensions.
    Very powerful for non-convex clusters (shapes that aren't circles/spheres).
    """

    name = "Spectral Clustering"
    description = "Graph-based clustering using eigenvalues for complex shapes"
    supports_n_clusters = True

    def __init__(self):
        super().__init__()
        self.params = {
            'n_clusters': 3,
            'affinity': 'rbf',
            'n_neighbors': 10,
            'gamma': 1.0,
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
                'description': 'Number of clusters'
            },
            {
                'name': 'affinity',
                'type': 'select',
                'default': 'rbf',
                'options': ['rbf', 'nearest_neighbors'],
                'description': 'How to construct the affinity matrix (Kernel)'
            },
            {
                'name': 'n_neighbors',
                'type': 'int',
                'default': 10,
                'min': 2,
                'max': 50,
                'description': 'Neighbors for nearest_neighbors affinity'
            },
            {
                'name': 'gamma',
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 10.0,
                'step': 0.1,
                'description': 'Kernel coefficient for rbf'
            }
        ]

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model = SpectralClustering(
            n_clusters=self.params['n_clusters'],
            affinity=self.params['affinity'],
            n_neighbors=self.params['n_neighbors'],
            gamma=self.params['gamma'],
            random_state=self.params['random_state'],
            n_jobs=-1
        )
        self.labels_ = self.model.fit_predict(X)
        return self.labels_