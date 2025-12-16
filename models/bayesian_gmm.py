"""Bayesian Gaussian Mixture Model."""

from typing import Dict, List, Any
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

from models.base import BaseClusterModel


class BayesianGMMModel(BaseClusterModel):
    """
    Bayesian Gaussian Mixture Model (Variational Inference).

    Unlike standard GMM, this infers the effective number of components
    from a prior distribution (Dirichlet Process). It can automatically
    set weights of unnecessary clusters to zero.
    
    Best for: Determining the 'true' number of clusters and probability of membership.
    """

    name = "Bayesian GMM"
    description = "Probabilistic model that infers effective cluster count using Variational Inference"
    supports_n_clusters = True  # We set an upper bound, model infers actual usage

    def __init__(self):
        super().__init__()
        self.params = {
            'n_components': 10,  # This is the upper bound, not fixed k
            'covariance_type': 'full',
            'weight_concentration_prior_type': 'dirichlet_process',
            'max_iter': 100,
            'random_state': 42
        }

    def get_param_config(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'n_components',
                'type': 'int',
                'default': 10,
                'min': 2,
                'max': 30,
                'description': 'Upper bound of components (model will deactivate unnecessary ones)'
            },
            {
                'name': 'covariance_type',
                'type': 'select',
                'default': 'full',
                'options': ['full', 'tied', 'diag', 'spherical'],
                'description': 'Covariance matrix type'
            },
            {
                'name': 'weight_concentration_prior_type',
                'type': 'select',
                'default': 'dirichlet_process',
                'options': ['dirichlet_process', 'dirichlet_distribution'],
                'description': 'Prior type (Dirichlet Process encourages fewer active clusters)'
            }
        ]

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model = BayesianGaussianMixture(
            n_components=self.params['n_components'],
            covariance_type=self.params['covariance_type'],
            weight_concentration_prior_type=self.params['weight_concentration_prior_type'],
            max_iter=self.params['max_iter'],
            random_state=self.params['random_state']
        )
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    def get_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Get probability of each sample belonging to each cluster."""
        if self.model is not None:
            return self.model.predict_proba(X)
        return np.array([])