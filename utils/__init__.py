"""Utility modules for ZE-Workbench."""

from .data_loader import load_file, download_df, get_variable_metadata
from .preprocessing import preprocess_features, encode_categorical, scale_features
from .metrics import calculate_cluster_metrics

# Binary data analysis (MCA, Tetrachoric Correlation)
from .binary_analysis import (
    tetrachoric_correlation_matrix,
    identify_redundant_features,
    perform_mca,
    get_binary_columns,
    binarize_dataframe
)

# Cluster profiling
from .cluster_profiling import (
    calculate_item_response_probabilities,
    calculate_relative_risk_ratios,
    generate_cluster_profiles,
    identify_poverty_archetypes,
    create_profile_heatmap_data,
    compare_clusters_with_external
)

# Validation
from .validation import (
    bootstrap_stability_analysis,
    external_validation,
    compare_model_fits,
    select_optimal_k,
    comprehensive_validation
)

__all__ = [
    # Data loading
    'load_file',
    'download_df',
    'get_variable_metadata',
    # Preprocessing
    'preprocess_features',
    'encode_categorical',
    'scale_features',
    # Metrics
    'calculate_cluster_metrics',
    # Binary analysis
    'tetrachoric_correlation_matrix',
    'identify_redundant_features',
    'perform_mca',
    'get_binary_columns',
    'binarize_dataframe',
    # Cluster profiling
    'calculate_item_response_probabilities',
    'calculate_relative_risk_ratios',
    'generate_cluster_profiles',
    'identify_poverty_archetypes',
    'create_profile_heatmap_data',
    'compare_clusters_with_external',
    # Validation
    'bootstrap_stability_analysis',
    'external_validation',
    'compare_model_fits',
    'select_optimal_k',
    'comprehensive_validation',
]
