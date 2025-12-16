"""
Clustering models package for ZE-Workbench.

This package provides a modular, extensible framework for clustering analysis.
Each model follows a consistent interface defined by BaseClusterModel.
"""

from models.base import BaseClusterModel
from models.kmeans import KMeansModel
from models.hierarchical import HierarchicalModel
from models.dbscan import DBSCANModel
from models.gaussian_mixture import GaussianMixtureModel
from models.optics import OPTICSModel
from models.spectral import SpectralModel
from models.runner import ModelRunner, ExperimentResult, create_experiment_configs

# Model registry for easy access
MODEL_REGISTRY = {
    'kmeans': KMeansModel,
    'hierarchical': HierarchicalModel,
    'dbscan': DBSCANModel,
    'gaussian_mixture': GaussianMixtureModel,
    'optics': OPTICSModel,
    'spectral': SpectralModel
}

# Display names for UI
MODEL_NAMES = {
    'kmeans': 'K-Means',
    'hierarchical': 'Hierarchical (Agglomerative)',
    'dbscan': 'DBSCAN',
    'gaussian_mixture': 'Gaussian Mixture (GMM)',
    'optics': 'OPTICS (Density)',
    'spectral': 'Spectral Clustering (Graph)'
}


def get_model(name: str) -> BaseClusterModel:
    """
    Factory function to get a model instance by name.

    Args:
        name: Model key ('kmeans', 'hierarchical', 'dbscan', 'gaussian_mixture', 'optics', 'spectral')

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
    'OPTICSModel',
    'SpectralModel',
    'ModelRunner',
    'ExperimentResult',
    'create_experiment_configs',
    'MODEL_REGISTRY',
    'MODEL_NAMES',
    'get_model',
    'list_models'
]
