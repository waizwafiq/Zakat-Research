"""Gaussian Mixture Model clustering."""

from typing import Dict, List, Any
import numpy as np
from sklearn.mixture import GaussianMixture

from models.base import BaseClusterModel


class GaussianMixtureModel(BaseClusterModel):
    """
    Gaussian Mixture Model (GMM) clustering.

    Probabilistic model assuming data comes from mixture of Gaussians.
    Good for soft clustering and elliptical cluster shapes.
    """

    name = "Gaussian Mixture"
    description = "Probabilistic clustering using mixture of Gaussian distributions"
    supports_n_clusters = True

    def __init__(self):
        super().__init__()
        self.params = {
            'n_components': 3,
            'covariance_type': 'full',
            'max_iter': 100,
            'n_init': 1,
            'random_state': 42
        }

    def get_param_config(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'n_components',
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 20,
                'description': 'Number of mixture components (clusters)'
            },
            {
                'name': 'covariance_type',
                'type': 'select',
                'default': 'full',
                'options': ['full', 'tied', 'diag', 'spherical'],
                'description': 'Covariance matrix type'
            },
            {
                'name': 'max_iter',
                'type': 'int',
                'default': 100,
                'min': 50,
                'max': 500,
                'description': 'Maximum EM iterations'
            },
            {
                'name': 'n_init',
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 10,
                'description': 'Number of initializations'
            }
        ]

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model = GaussianMixture(
            n_components=self.params['n_components'],
            covariance_type=self.params['covariance_type'],
            max_iter=self.params['max_iter'],
            n_init=self.params['n_init'],
            random_state=self.params['random_state']
        )
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    def get_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Get probability of each sample belonging to each cluster."""
        if self.model is not None:
            return self.model.predict_proba(X)
        return np.array([])

    def get_bic(self, X: np.ndarray) -> float:
        """Get Bayesian Information Criterion (lower is better)."""
        if self.model is not None:
            return self.model.bic(X)
        return float('inf')

    def get_aic(self, X: np.ndarray) -> float:
        """Get Akaike Information Criterion (lower is better)."""
        if self.model is not None:
            return self.model.aic(X)
        return float('inf')
