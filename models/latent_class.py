"""
Latent Class Analysis (LCA) Model.

LCA is a probabilistic model-based clustering approach specifically designed
for categorical/binary data. It assumes that observed responses are independent
conditional on the latent class membership.

Model:
    P(x_n) = Σ_k P(C_k) Π_d P(x_nd | C_k)

Where:
    - P(C_k) is the prior probability of belonging to cluster k
    - P(x_nd | C_k) is the probability of response in dimension d given class k
"""

import numpy as np
from typing import Dict, List, Any, Optional
from models.base import BaseClusterModel


class LatentClassModel(BaseClusterModel):
    """
    Latent Class Analysis for categorical/binary data clustering.

    Uses Expectation-Maximization (EM) algorithm to estimate:
    - Class membership probabilities (mixing proportions)
    - Item response probabilities for each class
    """

    name = "Latent Class Analysis (LCA)"
    description = "Probabilistic model for categorical/binary data clustering"
    supports_n_clusters = True

    def __init__(self):
        super().__init__()
        self.params = {
            'n_classes': 3,
            'max_iter': 100,
            'tol': 1e-6,
            'n_init': 10,
            'random_state': 42
        }
        self.class_probs_ = None  # P(C_k) - mixing proportions
        self.item_probs_ = None   # P(x_d=1 | C_k) - item response probabilities
        self.bic_ = None
        self.aic_ = None
        self.log_likelihood_ = None
        self.n_parameters_ = None

    def get_param_config(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'n_classes',
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 10,
                'description': 'Number of latent classes'
            },
            {
                'name': 'max_iter',
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 500,
                'description': 'Maximum EM iterations'
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
                'name': 'tol',
                'type': 'float',
                'default': 1e-6,
                'min': 1e-8,
                'max': 1e-3,
                'step': 1e-7,
                'description': 'Convergence tolerance'
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
            # Check if already binary
            unique_vals = np.unique(col[~np.isnan(col)])
            if len(unique_vals) <= 2:
                X_binary[:, j] = (col > np.min(unique_vals)).astype(float)
            else:
                # Binarize based on median
                median_val = np.nanmedian(col)
                X_binary[:, j] = (col > median_val).astype(float)
        return X_binary

    def _initialize_parameters(self, n_samples: int, n_features: int, k: int, rng: np.random.RandomState):
        """Initialize model parameters randomly."""
        # Class probabilities (uniform with noise)
        class_probs = rng.dirichlet(np.ones(k))

        # Item response probabilities with reasonable priors
        # Initialize between 0.1 and 0.9 to avoid numerical issues
        item_probs = rng.beta(2, 2, size=(k, n_features))
        item_probs = np.clip(item_probs, 0.01, 0.99)

        return class_probs, item_probs

    def _e_step(self, X: np.ndarray, class_probs: np.ndarray, item_probs: np.ndarray) -> np.ndarray:
        """
        E-step: Compute posterior probabilities of class membership.

        Returns:
            responsibilities: P(C_k | x_n) for each sample and class
        """
        n_samples, n_features = X.shape
        k = len(class_probs)

        # Log-likelihood for each class
        log_likelihood = np.zeros((n_samples, k))

        for c in range(k):
            # P(x | C_k) = Π_d P(x_d | C_k)
            # log P(x | C_k) = Σ_d log P(x_d | C_k)
            log_prob = np.zeros(n_samples)
            for d in range(n_features):
                p = item_probs[c, d]
                # P(x_d | C_k) = p^x_d * (1-p)^(1-x_d) for binary data
                log_prob += X[:, d] * np.log(p + 1e-10) + (1 - X[:, d]) * np.log(1 - p + 1e-10)

            log_likelihood[:, c] = np.log(class_probs[c] + 1e-10) + log_prob

        # Convert to responsibilities using log-sum-exp trick
        max_log = np.max(log_likelihood, axis=1, keepdims=True)
        log_likelihood_normalized = log_likelihood - max_log
        responsibilities = np.exp(log_likelihood_normalized)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True) + 1e-10

        return responsibilities

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        M-step: Update model parameters based on responsibilities.

        Returns:
            class_probs: Updated class probabilities
            item_probs: Updated item response probabilities
        """
        n_samples, n_features = X.shape
        k = responsibilities.shape[1]

        # Update class probabilities
        N_k = responsibilities.sum(axis=0)  # Effective class sizes
        class_probs = N_k / n_samples

        # Update item response probabilities
        item_probs = np.zeros((k, n_features))
        for c in range(k):
            for d in range(n_features):
                # Weighted mean of responses
                numerator = np.sum(responsibilities[:, c] * X[:, d])
                item_probs[c, d] = numerator / (N_k[c] + 1e-10)

        # Clip to avoid numerical issues
        item_probs = np.clip(item_probs, 0.01, 0.99)
        class_probs = np.clip(class_probs, 0.01, 0.99)
        class_probs /= class_probs.sum()

        return class_probs, item_probs

    def _compute_log_likelihood(self, X: np.ndarray, class_probs: np.ndarray, item_probs: np.ndarray) -> float:
        """Compute total log-likelihood of the data."""
        n_samples, n_features = X.shape
        k = len(class_probs)

        total_log_likelihood = 0.0

        for n in range(n_samples):
            sample_likelihood = 0.0
            for c in range(k):
                class_likelihood = class_probs[c]
                for d in range(n_features):
                    p = item_probs[c, d]
                    if X[n, d] == 1:
                        class_likelihood *= p
                    else:
                        class_likelihood *= (1 - p)
                sample_likelihood += class_likelihood

            total_log_likelihood += np.log(sample_likelihood + 1e-300)

        return total_log_likelihood

    def _fit_single(self, X: np.ndarray, random_state: int) -> Dict:
        """Run EM algorithm once with a given random state."""
        n_samples, n_features = X.shape
        k = self.params['n_classes']
        rng = np.random.RandomState(random_state)

        # Initialize
        class_probs, item_probs = self._initialize_parameters(n_samples, n_features, k, rng)

        prev_log_likelihood = -np.inf

        for iteration in range(self.params['max_iter']):
            # E-step
            responsibilities = self._e_step(X, class_probs, item_probs)

            # M-step
            class_probs, item_probs = self._m_step(X, responsibilities)

            # Check convergence
            log_likelihood = self._compute_log_likelihood(X, class_probs, item_probs)

            if abs(log_likelihood - prev_log_likelihood) < self.params['tol']:
                break

            prev_log_likelihood = log_likelihood

        return {
            'class_probs': class_probs,
            'item_probs': item_probs,
            'log_likelihood': log_likelihood,
            'responsibilities': responsibilities,
            'n_iterations': iteration + 1
        }

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit LCA model and return cluster labels.

        Args:
            X: Feature array (will be binarized internally)

        Returns:
            Array of cluster labels (0 to k-1)
        """
        # Binarize data
        X_binary = self._binarize(X)
        n_samples, n_features = X_binary.shape
        k = self.params['n_classes']

        # Run multiple initializations
        best_result = None
        best_log_likelihood = -np.inf

        base_random_state = self.params.get('random_state', 42)

        for init in range(self.params['n_init']):
            result = self._fit_single(X_binary, base_random_state + init)

            if result['log_likelihood'] > best_log_likelihood:
                best_log_likelihood = result['log_likelihood']
                best_result = result

        # Store results
        self.class_probs_ = best_result['class_probs']
        self.item_probs_ = best_result['item_probs']
        self.log_likelihood_ = best_result['log_likelihood']

        # Calculate BIC and AIC
        n_parameters = (k - 1) + k * n_features  # class probs + item probs
        self.n_parameters_ = n_parameters
        self.bic_ = -2 * self.log_likelihood_ + n_parameters * np.log(n_samples)
        self.aic_ = -2 * self.log_likelihood_ + 2 * n_parameters

        # Entropy (classification certainty)
        responsibilities = best_result['responsibilities']
        entropy = -np.sum(responsibilities * np.log(responsibilities + 1e-10)) / n_samples
        max_entropy = np.log(k)
        self.entropy_ = 1 - (entropy / max_entropy) if max_entropy > 0 else 0

        # Assign labels based on maximum posterior probability
        self.labels_ = np.argmax(responsibilities, axis=1)

        return self.labels_

    def get_item_response_probabilities(self) -> np.ndarray:
        """
        Get item response probabilities P(x_d=1 | C_k).

        Returns:
            Array of shape (n_classes, n_features)
        """
        return self.item_probs_

    def get_class_probabilities(self) -> np.ndarray:
        """
        Get class membership probabilities P(C_k).

        Returns:
            Array of shape (n_classes,)
        """
        return self.class_probs_

    def get_model_fit_statistics(self) -> Dict:
        """
        Get model fit statistics for model selection.

        Returns:
            Dict with BIC, AIC, log-likelihood, entropy
        """
        return {
            'log_likelihood': self.log_likelihood_,
            'bic': self.bic_,
            'aic': self.aic_,
            'entropy': self.entropy_,
            'n_parameters': self.n_parameters_
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class membership probabilities for new data.

        Args:
            X: Feature array

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        X_binary = self._binarize(X)
        return self._e_step(X_binary, self.class_probs_, self.item_probs_)
