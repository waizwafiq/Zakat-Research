"""DBSCAN clustering model."""

from typing import Dict, List, Any
import numpy as np
from sklearn.cluster import DBSCAN

from models.base import BaseClusterModel


class DBSCANModel(BaseClusterModel):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Finds clusters based on density, automatically detecting cluster count.
    Good for discovering clusters of arbitrary shape and identifying outliers.
    """

    name = "DBSCAN"
    description = "Density-based clustering that finds arbitrarily shaped clusters"
    supports_n_clusters = False  # DBSCAN determines clusters automatically

    def __init__(self):
        super().__init__()
        self.params = {
            'eps': 0.5,
            'min_samples': 5,
            'metric': 'euclidean'
        }

    def get_param_config(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'eps',
                'type': 'float',
                'default': 0.5,
                'min': 0.1,
                'max': 10.0,
                'step': 0.1,
                'description': 'Maximum distance between samples (epsilon)'
            },
            {
                'name': 'min_samples',
                'type': 'int',
                'default': 5,
                'min': 2,
                'max': 50,
                'description': 'Minimum samples in neighborhood'
            },
            {
                'name': 'metric',
                'type': 'select',
                'default': 'euclidean',
                'options': ['euclidean', 'manhattan', 'cosine'],
                'description': 'Distance metric'
            }
        ]

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model = DBSCAN(
            eps=self.params['eps'],
            min_samples=self.params['min_samples'],
            metric=self.params['metric']
        )
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    def get_core_sample_indices(self) -> np.ndarray:
        """Get indices of core samples."""
        if self.model is not None:
            return self.model.core_sample_indices_
        return np.array([])
