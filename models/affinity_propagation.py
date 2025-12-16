"""Affinity Propagation clustering model."""

from typing import Dict, List, Any
import numpy as np
from sklearn.cluster import AffinityPropagation

from models.base import BaseClusterModel


class AffinityPropagationModel(BaseClusterModel):
    """
    Affinity Propagation.

    Finds "exemplars" (representative data points) by passing messages between pairs of samples.
    Does NOT require specifying the number of clusters.
    
    Best for: Finding representative "Archetypes" in small datasets (like your 400 rows).
    """

    name = "Affinity Propagation"
    description = "Exemplar-based clustering that finds representative data points automatically"
    supports_n_clusters = False  # Determined by preference/damping

    def __init__(self):
        super().__init__()
        self.params = {
            'damping': 0.5,
            'max_iter': 200,
            'random_state': 42
        }

    def get_param_config(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'damping',
                'type': 'float',
                'default': 0.5,
                'min': 0.5,
                'max': 0.99,
                'step': 0.01,
                'description': 'Damping factor (0.5 to 1.0) to avoid numerical oscillations'
            },
            {
                'name': 'max_iter',
                'type': 'int',
                'default': 200,
                'min': 50,
                'max': 1000,
                'description': 'Maximum iterations'
            }
        ]

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model = AffinityPropagation(
            damping=self.params['damping'],
            max_iter=self.params['max_iter'],
            random_state=self.params['random_state']
        )
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    def get_exemplars(self) -> np.ndarray:
        """Get indices of the exemplar instances."""
        if self.model is not None:
            return self.model.cluster_centers_indices_
        return np.array([])