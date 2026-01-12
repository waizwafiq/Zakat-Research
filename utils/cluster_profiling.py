"""
Cluster Profiling Module for MPI Data.

This module provides advanced cluster characterization methods:
- Item Response Probabilities
- Relative Risk Ratios
- Deprivation Profiles ("Faces of Poverty")
- Cluster interpretation and naming
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def calculate_item_response_probabilities(X: np.ndarray, labels: np.ndarray,
                                          feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate Item Response Probabilities P(x_d=1 | C_k).

    For each cluster k and feature d, compute the probability that
    a household in cluster k is deprived in dimension d.

    Args:
        X: Binary feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        feature_names: Optional list of feature names

    Returns:
        DataFrame with clusters as rows and features as columns
    """
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]

    # Binarize if needed
    X_binary = np.zeros_like(X)
    for j in range(n_features):
        col = X[:, j]
        unique_vals = np.unique(col[~np.isnan(col)])
        if len(unique_vals) <= 2:
            X_binary[:, j] = (col > np.min(unique_vals)).astype(float)
        else:
            X_binary[:, j] = (col > np.median(col)).astype(float)

    unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
    n_clusters = len(unique_labels)

    # Calculate probabilities
    probs = np.zeros((n_clusters, n_features))

    for i, k in enumerate(unique_labels):
        cluster_mask = labels == k
        cluster_data = X_binary[cluster_mask]
        probs[i] = cluster_data.mean(axis=0)

    # Create DataFrame
    cluster_names = [f'Cluster {k}' for k in unique_labels]
    result = pd.DataFrame(probs, index=cluster_names, columns=feature_names)

    return result


def calculate_relative_risk_ratios(X: np.ndarray, labels: np.ndarray,
                                   feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate Relative Risk Ratios for each cluster and feature.

    RR_kd = P(x_d=1 | C_k) / P(x_d=1 | Population)

    Interpretation:
    - RR > 1: Cluster k has HIGHER deprivation rate than population
    - RR < 1: Cluster k has LOWER deprivation rate than population
    - RR = 1: Same as population average

    Args:
        X: Binary feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        feature_names: Optional list of feature names

    Returns:
        DataFrame with clusters as rows and features as columns
    """
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]

    # Binarize if needed
    X_binary = np.zeros_like(X)
    for j in range(n_features):
        col = X[:, j]
        unique_vals = np.unique(col[~np.isnan(col)])
        if len(unique_vals) <= 2:
            X_binary[:, j] = (col > np.min(unique_vals)).astype(float)
        else:
            X_binary[:, j] = (col > np.median(col)).astype(float)

    # Population rates
    population_rates = X_binary.mean(axis=0)

    unique_labels = np.unique(labels[labels >= 0])
    n_clusters = len(unique_labels)

    # Calculate relative risk
    rr = np.zeros((n_clusters, n_features))

    for i, k in enumerate(unique_labels):
        cluster_mask = labels == k
        cluster_data = X_binary[cluster_mask]
        cluster_rates = cluster_data.mean(axis=0)

        # Avoid division by zero
        rr[i] = cluster_rates / (population_rates + 1e-10)

    # Create DataFrame
    cluster_names = [f'Cluster {k}' for k in unique_labels]
    result = pd.DataFrame(rr, index=cluster_names, columns=feature_names)

    return result


def generate_cluster_profiles(X: np.ndarray, labels: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              rr_threshold: float = 1.5) -> Dict:
    """
    Generate comprehensive cluster profiles ("Faces of Poverty").

    Args:
        X: Binary feature matrix
        labels: Cluster labels
        feature_names: Feature names
        rr_threshold: RR threshold for highlighting (default: 1.5)

    Returns:
        Dict with:
            - 'item_response_probs': Item response probability matrix
            - 'relative_risks': Relative risk ratio matrix
            - 'cluster_sizes': Number of samples per cluster
            - 'cluster_proportions': Proportion of samples per cluster
            - 'high_risk_features': Features with RR > threshold per cluster
            - 'low_risk_features': Features with RR < 1/threshold per cluster
            - 'cluster_descriptions': Textual descriptions of each cluster
    """
    n_samples, n_features = X.shape

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]

    # Calculate probabilities and RR
    irp = calculate_item_response_probabilities(X, labels, feature_names)
    rr = calculate_relative_risk_ratios(X, labels, feature_names)

    # Cluster sizes and proportions
    unique_labels = np.unique(labels[labels >= 0])
    cluster_sizes = {}
    cluster_proportions = {}

    for k in unique_labels:
        size = np.sum(labels == k)
        cluster_sizes[f'Cluster {k}'] = size
        cluster_proportions[f'Cluster {k}'] = size / n_samples

    # Identify high/low risk features
    high_risk_features = {}
    low_risk_features = {}
    cluster_descriptions = {}

    for i, k in enumerate(unique_labels):
        cluster_name = f'Cluster {k}'

        # High risk (RR > threshold)
        high_mask = rr.loc[cluster_name] > rr_threshold
        high_risk_features[cluster_name] = rr.loc[cluster_name][high_mask].sort_values(ascending=False).to_dict()

        # Low risk (RR < 1/threshold)
        low_mask = rr.loc[cluster_name] < (1 / rr_threshold)
        low_risk_features[cluster_name] = rr.loc[cluster_name][low_mask].sort_values().to_dict()

        # Generate description
        description = _generate_cluster_description(
            cluster_name,
            irp.loc[cluster_name],
            rr.loc[cluster_name],
            cluster_proportions[cluster_name],
            rr_threshold
        )
        cluster_descriptions[cluster_name] = description

    return {
        'item_response_probs': irp,
        'relative_risks': rr,
        'cluster_sizes': cluster_sizes,
        'cluster_proportions': cluster_proportions,
        'high_risk_features': high_risk_features,
        'low_risk_features': low_risk_features,
        'cluster_descriptions': cluster_descriptions
    }


def _generate_cluster_description(cluster_name: str, irp: pd.Series,
                                  rr: pd.Series, proportion: float,
                                  threshold: float) -> str:
    """
    Generate a human-readable description of a cluster.

    This attempts to identify the "archetype" of poverty represented
    by the cluster based on the relative risk patterns.
    """
    high_rr = rr[rr > threshold].sort_values(ascending=False)
    low_rr = rr[rr < 1/threshold].sort_values()

    # Determine archetype
    mean_rr = rr.mean()
    high_count = len(high_rr)
    low_count = len(low_rr)

    description_parts = [f"({proportion*100:.1f}% of population)"]

    if mean_rr > 1.5 and high_count >= len(rr) * 0.5:
        archetype = "Deeply Deprived"
        description_parts.append("High deprivation across most indicators.")
    elif mean_rr < 0.7 and low_count >= len(rr) * 0.5:
        archetype = "Low Deprivation"
        description_parts.append("Low deprivation rates across most indicators.")
    elif high_count > 0 and low_count > 0:
        archetype = "Mixed Profile"
        description_parts.append("Mixed deprivation pattern.")
    else:
        archetype = "Moderate"
        description_parts.append("Near-average deprivation levels.")

    # Add specific high-risk indicators
    if len(high_rr) > 0:
        top_risks = list(high_rr.head(3).index)
        description_parts.append(f"High risk in: {', '.join(top_risks)}.")

    # Add specific low-risk indicators
    if len(low_rr) > 0:
        top_protections = list(low_rr.head(3).index)
        description_parts.append(f"Protected from: {', '.join(top_protections)}.")

    return f"{archetype}: " + " ".join(description_parts)


def identify_poverty_archetypes(profiles: Dict, feature_groups: Optional[Dict] = None) -> Dict:
    """
    Identify poverty archetypes based on cluster profiles.

    Hypothesized archetypes for MPI data:
    1. The Deeply Deprived: High across all indicators
    2. The Income Poor Only: Deprived in income but not services
    3. The Infrastructurally Deprived: Poor services despite income
    4. The Maqasid Vulnerable: Social/spiritual deprivation

    Args:
        profiles: Output from generate_cluster_profiles
        feature_groups: Optional dict mapping group names to feature lists
                       e.g., {'Income': ['PENDAPAT'], 'Services': ['TIDAKDAP', 'MENGGUNA']}

    Returns:
        Dict mapping cluster names to archetype names
    """
    rr = profiles['relative_risks']
    archetypes = {}

    for cluster in rr.index:
        cluster_rr = rr.loc[cluster]
        mean_rr = cluster_rr.mean()

        # Simple heuristic archetype assignment
        if mean_rr > 1.5:
            archetypes[cluster] = "Deeply Deprived"
        elif mean_rr < 0.6:
            archetypes[cluster] = "Low Deprivation"
        elif mean_rr > 1.0:
            archetypes[cluster] = "Moderately Deprived"
        else:
            archetypes[cluster] = "Near Average"

    return archetypes


def create_profile_heatmap_data(profiles: Dict, metric: str = 'relative_risks') -> pd.DataFrame:
    """
    Prepare data for profile visualization heatmap.

    Args:
        profiles: Output from generate_cluster_profiles
        metric: 'relative_risks' or 'item_response_probs'

    Returns:
        DataFrame suitable for heatmap visualization
    """
    if metric == 'relative_risks':
        data = profiles['relative_risks']
    else:
        data = profiles['item_response_probs']

    return data.T  # Transpose for features as rows, clusters as columns


def compare_clusters_with_external(labels: np.ndarray, external_var: np.ndarray,
                                   external_name: str = 'External') -> pd.DataFrame:
    """
    Cross-tabulate clusters with an external variable for validation.

    Args:
        labels: Cluster labels
        external_var: External categorical variable (e.g., urban/rural)
        external_name: Name for the external variable

    Returns:
        DataFrame crosstab with percentages
    """
    # Create crosstab
    cluster_series = pd.Series(labels, name='Cluster')
    external_series = pd.Series(external_var, name=external_name)

    # Exclude noise points
    mask = labels >= 0
    cluster_series = cluster_series[mask]
    external_series = external_series[mask]

    # Create crosstab
    ct = pd.crosstab(cluster_series, external_series, normalize='index') * 100

    return ct.round(2)
