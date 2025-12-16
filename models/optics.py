"""OPTICS clustering model."""

from typing import Dict, List, Any
import numpy as np
from sklearn.cluster import OPTICS

from models.base import BaseClusterModel


class OPTICSModel(BaseClusterModel):
    """
    OPTICS (Ordering Points To Identify the Clustering Structure).

    An extension of DBSCAN that handles varying densities better.
    Instead of a fixed epsilon, it builds a reachability graph.
    Good for complex real-world data with clusters of different densities.
    """

    name = "OPTICS"
    description = "Advanced density-based clustering for varying densities"
    supports_n_clusters = False  # Determines clusters automatically

    def __init__(self):
        super().__init__()
        self.params = {
            'min_samples': 5,
            'metric': 'minkowski',
            'xi': 0.05,
            'min_cluster_size': 0.05  # float = fraction of samples
        }

    def get_param_config(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'min_samples',
                'type': 'int',
                'default': 5,
                'min': 2,
                'max': 50,
                'description': 'Minimum samples to form a core point'
            },
            {
                'name': 'xi',
                'type': 'float',
                'default': 0.05,
                'min': 0.01,
                'max': 0.5,
                'step': 0.01,
                'description': 'Steepness threshold for cluster extraction'
            },
            {
                'name': 'min_cluster_size',
                'type': 'float',
                'default': 0.05,
                'min': 0.01,
                'max': 0.5,
                'step': 0.01,
                'description': 'Min fraction of samples to constitute a cluster'
            },
            {
                'name': 'metric',
                'type': 'select',
                'default': 'minkowski',
                'options': ['minkowski', 'euclidean', 'manhattan', 'cosine'],
                'description': 'Distance metric'
            }
        ]

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        # OPTICS allows min_cluster_size to be float (fraction) or int (count)
        # We ensure valid types here
        self.model = OPTICS(
            min_samples=self.params['min_samples'],
            metric=self.params['metric'],
            xi=self.params['xi'],
            min_cluster_size=self.params['min_cluster_size'],
            n_jobs=-1  # Use parallel processing
        )
        self.labels_ = self.model.fit_predict(X)
        return self.labels_