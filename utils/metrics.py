"""Clustering evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def calculate_cluster_metrics(X, labels):
    """
    Calculate comprehensive clustering evaluation metrics.

    Args:
        X: Feature array (scaled)
        labels: Cluster labels

    Returns:
        dict with metrics:
            - silhouette: Silhouette Score (-1 to 1, higher is better)
            - davies_bouldin: Davies-Bouldin Index (lower is better)
            - calinski_harabasz: Calinski-Harabasz Index (higher is better)
            - n_clusters: Number of clusters found
            - n_noise: Number of noise points (for DBSCAN)
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    # Need at least 2 clusters for meaningful metrics
    if n_clusters < 2:
        return {
            'silhouette': 0.0,
            'davies_bouldin': float('inf'),
            'calinski_harabasz': 0.0,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'valid': False
        }

    # Filter out noise points for metric calculation
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    try:
        sil = silhouette_score(X_valid, labels_valid)
    except Exception:
        sil = 0.0

    try:
        db = davies_bouldin_score(X_valid, labels_valid)
    except Exception:
        db = float('inf')

    try:
        ch = calinski_harabasz_score(X_valid, labels_valid)
    except Exception:
        ch = 0.0

    return {
        'silhouette': round(sil, 4),
        'davies_bouldin': round(db, 4),
        'calinski_harabasz': round(ch, 4),
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'valid': True
    }


def format_metrics_for_display(metrics):
    """
    Format metrics dictionary for display in Streamlit.

    Args:
        metrics: dict from calculate_cluster_metrics

    Returns:
        dict with formatted strings
    """
    return {
        'Silhouette Score': f"{metrics['silhouette']:.4f}",
        'Davies-Bouldin Index': f"{metrics['davies_bouldin']:.4f}",
        'Calinski-Harabasz Index': f"{metrics['calinski_harabasz']:.2f}",
        'Clusters Found': metrics['n_clusters'],
        'Noise Points': metrics['n_noise']
    }
