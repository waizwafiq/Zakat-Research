"""
Clustering models package for PySPSS.

This package provides a modular, extensible framework for clustering analysis.
Each model follows a consistent interface defined by BaseClusterModel.
"""

from models.base import BaseClusterModel
from models.kmeans import KMeansModel
from models.hierarchical import HierarchicalModel
from models.dbscan import DBSCANModel
from models.gaussian_mixture import GaussianMixtureModel
from models.runner import ModelRunner, ExperimentResult, create_experiment_configs

# Model registry for easy access
MODEL_REGISTRY = {
    'kmeans': KMeansModel,
    'hierarchical': HierarchicalModel,
    'dbscan': DBSCANModel,
    'gaussian_mixture': GaussianMixtureModel
}

# Display names for UI
MODEL_NAMES = {
    'kmeans': 'K-Means',
    'hierarchical': 'Hierarchical (Agglomerative)',
    'dbscan': 'DBSCAN',
    'gaussian_mixture': 'Gaussian Mixture (GMM)'
}


def get_model(name: str) -> BaseClusterModel:
    """
    Factory function to get a model instance by name.

    Args:
        name: Model key ('kmeans', 'hierarchical', 'dbscan', 'gaussian_mixture')

    Returns:
        Instance of the requested model
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()


def list_models():
    """Get list of available model names."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    'BaseClusterModel',
    'KMeansModel',
    'HierarchicalModel',
    'DBSCANModel',
    'GaussianMixtureModel',
    'ModelRunner',
    'ExperimentResult',
    'create_experiment_configs',
    'MODEL_REGISTRY',
    'MODEL_NAMES',
    'get_model',
    'list_models'
]
