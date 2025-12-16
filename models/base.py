"""Base class for all clustering models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np

from utils.metrics import calculate_cluster_metrics


class BaseClusterModel(ABC):
    """
    Abstract base class for clustering models.

    All clustering models should inherit from this class and implement
    the required methods for consistent interface.
    """

    name: str = "Base Model"
    description: str = "Base clustering model"
    supports_n_clusters: bool = True  # Whether model accepts n_clusters param

    def __init__(self):
        self.model = None
        self.labels_ = None
        self.params = {}

    @abstractmethod
    def get_param_config(self) -> List[Dict[str, Any]]:
        """
        Get parameter configuration for UI rendering.

        Returns:
            List of parameter configs, each with:
                - name: parameter name
                - type: 'int', 'float', 'select'
                - default: default value
                - min/max: for numeric types
                - options: for select type
                - description: help text
        """
        pass

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """Set model parameters."""
        pass

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and return cluster labels.

        Args:
            X: Feature array (scaled)

        Returns:
            Array of cluster labels
        """
        pass

    def get_metrics(self, X: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate clustering metrics.

        Args:
            X: Feature array (scaled)
            labels: Cluster labels (uses self.labels_ if None)

        Returns:
            Dict of metrics
        """
        if labels is None:
            labels = self.labels_
        return calculate_cluster_metrics(X, labels)

    def get_params(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self.params.copy()

    def get_params_string(self) -> str:
        """Get parameters as formatted string for display."""
        return ", ".join(f"{k}={v}" for k, v in self.params.items())

    @classmethod
    def get_info(cls) -> Dict[str, str]:
        """Get model info for display."""
        return {
            'name': cls.name,
            'description': cls.description,
            'supports_n_clusters': cls.supports_n_clusters
        }
