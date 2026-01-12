"""
Clustering models package for ZE-Workbench.

This package provides a modular, extensible framework for clustering analysis.
Each model follows a consistent interface defined by BaseClusterModel.

Includes specialized models for binary/categorical data:
- Latent Class Analysis (LCA) - probabilistic gold standard
- K-Modes with Jaccard distance - distance-based for categorical data
- Hierarchical with Jaccard - agglomerative for binary data
"""

from models.base import BaseClusterModel
from models.kmeans import KMeansModel
from models.hierarchical import HierarchicalModel
from models.dbscan import DBSCANModel
from models.gaussian_mixture import GaussianMixtureModel
from models.optics import OPTICSModel
from models.spectral import SpectralModel
from models.affinity_propagation import AffinityPropagationModel
from models.bayesian_gmm import BayesianGMMModel
from models.latent_class import LatentClassModel
from models.kmodes import KModesModel, HierarchicalJaccardModel
from models.runner import ModelRunner, ExperimentResult, create_experiment_configs

# Model registry for easy access
MODEL_REGISTRY = {
    'kmeans': KMeansModel,
    'hierarchical': HierarchicalModel,
    'dbscan': DBSCANModel,
    'gaussian_mixture': GaussianMixtureModel,
    'optics': OPTICSModel,
    'spectral': SpectralModel,
    'affinity_propagation': AffinityPropagationModel,
    'bayesian_gmm': BayesianGMMModel,
    'latent_class': LatentClassModel,
    'kmodes': KModesModel,
    'hierarchical_jaccard': HierarchicalJaccardModel
}

# Display names for UI
MODEL_NAMES = {
    'kmeans': 'K-Means',
    'hierarchical': 'Hierarchical (Agglomerative)',
    'dbscan': 'DBSCAN',
    'gaussian_mixture': 'Gaussian Mixture (GMM)',
    'optics': 'OPTICS (Density)',
    'spectral': 'Spectral Clustering (Graph)',
    'affinity_propagation': 'Affinity Propagation',
    'bayesian_gmm': 'Bayesian GMM (Variational)',
    'latent_class': 'Latent Class Analysis (LCA)',
    'kmodes': 'K-Modes (Categorical/Binary)',
    'hierarchical_jaccard': 'Hierarchical (Jaccard Distance)'
}

# Models specifically designed for binary/categorical data
BINARY_DATA_MODELS = ['latent_class', 'kmodes', 'hierarchical_jaccard']

# =============================================================================
# DISABLED MODELS
# =============================================================================
# The following models are DISABLED for MPI binary data analysis.
# Reason: K-Means and similar algorithms assume continuous, normally distributed
# variables with Euclidean geometry, which is mathematically inappropriate for
# binary deprivation indicators. Use LCA, K-Modes, or Hierarchical Jaccard instead.
#
# To re-enable, remove the model key from DISABLED_MODELS list.
# =============================================================================
DISABLED_MODELS = [
    'kmeans',           # Assumes continuous data, Euclidean distance
    'hierarchical',     # Uses Euclidean distance (use hierarchical_jaccard instead)
    'dbscan',           # Density-based with Euclidean distance
    'gaussian_mixture', # Assumes Gaussian distributions
    'optics',           # Density-based with Euclidean distance
    'spectral',         # Graph-based but typically uses Euclidean affinity
    'affinity_propagation',  # Uses Euclidean-based similarity
    'bayesian_gmm',     # Assumes Gaussian distributions
]

# Enabled models for UI (only appropriate for binary/categorical MPI data)
ENABLED_MODELS = [k for k in MODEL_REGISTRY.keys() if k not in DISABLED_MODELS]

# Display names for enabled models only
ENABLED_MODEL_NAMES = {k: v for k, v in MODEL_NAMES.items() if k in ENABLED_MODELS}


def get_model(name: str) -> BaseClusterModel:
    """
    Factory function to get a model instance by name.

    Args:
        name: Model key ('kmeans', 'hierarchical', 'dbscan', 'gaussian_mixture',
              'optics', 'spectral', 'affinity_propagation', 'bayesian_gmm')

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
    'AffinityPropagationModel',
    'BayesianGMMModel',
    'LatentClassModel',
    'KModesModel',
    'HierarchicalJaccardModel',
    'ModelRunner',
    'ExperimentResult',
    'create_experiment_configs',
    'MODEL_REGISTRY',
    'MODEL_NAMES',
    'BINARY_DATA_MODELS',
    'DISABLED_MODELS',
    'ENABLED_MODELS',
    'ENABLED_MODEL_NAMES',
    'get_model',
    'list_models'
]
