"""
Cluster Validation Module.

This module provides comprehensive validation for clustering results:
- Bootstrap Stability Analysis with Adjusted Rand Index (ARI)
- External Validation (cross-tabulation with known variables)
- Model Selection Criteria (BIC for LCA)
- Cluster Quality Metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings


def bootstrap_stability_analysis(X: np.ndarray, clustering_func: Callable,
                                 n_bootstrap: int = 100,
                                 sample_ratio: float = 0.8,
                                 random_state: int = 42) -> Dict:
    """
    Perform Bootstrap Stability Analysis on clustering.

    Resamples the dataset n_bootstrap times and measures the Adjusted Rand Index
    between the original clusters and the bootstrap clusters.

    Stable clusters should have ARI > 0.7 across bootstrap samples.

    Args:
        X: Feature matrix (n_samples, n_features)
        clustering_func: Function that takes X and returns labels
                        e.g., lambda x: model.fit_predict(x)
        n_bootstrap: Number of bootstrap iterations (default: 100)
        sample_ratio: Proportion of data to sample (default: 0.8)
        random_state: Random seed for reproducibility

    Returns:
        Dict with:
            - 'mean_ari': Mean ARI across bootstrap samples
            - 'std_ari': Standard deviation of ARI
            - 'median_ari': Median ARI
            - 'min_ari': Minimum ARI
            - 'max_ari': Maximum ARI
            - 'ari_values': List of all ARI values
            - 'stability_grade': 'Excellent', 'Good', 'Fair', or 'Poor'
            - 'is_stable': Boolean (True if mean_ari > 0.7)
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    sample_size = int(n_samples * sample_ratio)

    # Get original labels
    original_labels = clustering_func(X)

    ari_values = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = rng.choice(n_samples, size=sample_size, replace=True)
        X_bootstrap = X[indices]

        try:
            # Cluster bootstrap sample
            bootstrap_labels = clustering_func(X_bootstrap)

            # Get labels for the sampled indices from original clustering
            original_subset = original_labels[indices]

            # Calculate ARI between original and bootstrap clustering
            ari = adjusted_rand_score(original_subset, bootstrap_labels)
            ari_values.append(ari)

        except Exception as e:
            # Skip failed bootstrap iterations
            warnings.warn(f"Bootstrap iteration {i} failed: {e}")
            continue

    if len(ari_values) == 0:
        return {
            'mean_ari': 0.0,
            'std_ari': 0.0,
            'median_ari': 0.0,
            'min_ari': 0.0,
            'max_ari': 0.0,
            'ari_values': [],
            'stability_grade': 'Failed',
            'is_stable': False
        }

    ari_array = np.array(ari_values)
    mean_ari = np.mean(ari_array)
    std_ari = np.std(ari_array)
    median_ari = np.median(ari_array)
    min_ari = np.min(ari_array)
    max_ari = np.max(ari_array)

    # Determine stability grade
    if mean_ari >= 0.9:
        grade = 'Excellent'
    elif mean_ari >= 0.7:
        grade = 'Good'
    elif mean_ari >= 0.5:
        grade = 'Fair'
    else:
        grade = 'Poor'

    return {
        'mean_ari': mean_ari,
        'std_ari': std_ari,
        'median_ari': median_ari,
        'min_ari': min_ari,
        'max_ari': max_ari,
        'ari_values': ari_values,
        'stability_grade': grade,
        'is_stable': mean_ari >= 0.7,
        'n_successful_iterations': len(ari_values)
    }


def external_validation(labels: np.ndarray, external_var: np.ndarray,
                        var_name: str = 'External') -> Dict:
    """
    Perform external validation by comparing clusters with a known variable.

    Args:
        labels: Cluster labels
        external_var: External categorical variable (e.g., Urban/Rural, Bandar/Luar Bandar)
        var_name: Name of the external variable

    Returns:
        Dict with:
            - 'crosstab': Cross-tabulation of clusters vs external variable
            - 'crosstab_pct': Cross-tab as percentages (row-normalized)
            - 'chi_square': Chi-square statistic
            - 'p_value': P-value for chi-square test
            - 'cramers_v': Cramer's V (effect size)
            - 'nmi': Normalized Mutual Information
    """
    from scipy import stats

    # Exclude noise points
    mask = labels >= 0
    labels_clean = labels[mask]
    external_clean = external_var[mask]

    # Create crosstab
    ct = pd.crosstab(
        pd.Series(labels_clean, name='Cluster'),
        pd.Series(external_clean, name=var_name)
    )

    # Percentage crosstab (row-normalized)
    ct_pct = pd.crosstab(
        pd.Series(labels_clean, name='Cluster'),
        pd.Series(external_clean, name=var_name),
        normalize='index'
    ) * 100

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(ct)

    # Cramer's V (effect size)
    n = ct.values.sum()
    min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    # Normalized Mutual Information
    nmi = normalized_mutual_info_score(labels_clean, external_clean)

    return {
        'crosstab': ct,
        'crosstab_pct': ct_pct.round(2),
        'chi_square': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'nmi': nmi,
        'degrees_of_freedom': dof
    }


def compare_model_fits(models_results: List[Dict]) -> pd.DataFrame:
    """
    Compare multiple LCA or clustering model fits.

    Args:
        models_results: List of dicts, each containing:
            - 'name': Model name/description
            - 'n_clusters': Number of clusters
            - 'bic': BIC (optional)
            - 'aic': AIC (optional)
            - 'log_likelihood': Log-likelihood (optional)
            - 'silhouette': Silhouette score (optional)
            - 'stability_ari': Bootstrap ARI (optional)

    Returns:
        DataFrame comparing all models
    """
    comparison_data = []

    for result in models_results:
        row = {
            'Model': result.get('name', 'Unknown'),
            'K': result.get('n_clusters', 'N/A'),
            'BIC': result.get('bic', np.nan),
            'AIC': result.get('aic', np.nan),
            'Log-Likelihood': result.get('log_likelihood', np.nan),
            'Silhouette': result.get('silhouette', np.nan),
            'Davies-Bouldin': result.get('davies_bouldin', np.nan),
            'Stability (ARI)': result.get('stability_ari', np.nan)
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Sort by BIC if available
    if 'BIC' in df.columns and not df['BIC'].isna().all():
        df = df.sort_values('BIC')

    return df


def select_optimal_k(X: np.ndarray, clustering_class: Any,
                     k_range: range = range(2, 9),
                     criterion: str = 'bic',
                     random_state: int = 42) -> Dict:
    """
    Select optimal number of clusters using information criteria.

    Args:
        X: Feature matrix
        clustering_class: Clustering model class (e.g., LatentClassModel)
        k_range: Range of cluster numbers to try
        criterion: Selection criterion ('bic', 'aic', 'silhouette')
        random_state: Random seed

    Returns:
        Dict with:
            - 'optimal_k': Best number of clusters
            - 'results': DataFrame with all results
            - 'best_model': Fitted model with optimal k
    """
    results = []

    for k in k_range:
        try:
            model = clustering_class()
            model.set_params(n_classes=k, random_state=random_state)
            labels = model.fit_predict(X)

            # Get fit statistics
            if hasattr(model, 'get_model_fit_statistics'):
                stats = model.get_model_fit_statistics()
                bic = stats.get('bic', np.nan)
                aic = stats.get('aic', np.nan)
                ll = stats.get('log_likelihood', np.nan)
                entropy = stats.get('entropy', np.nan)
            else:
                bic = aic = ll = entropy = np.nan

            # Calculate silhouette if possible
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                try:
                    sil = silhouette_score(X, labels)
                except:
                    sil = np.nan
            else:
                sil = np.nan

            results.append({
                'k': k,
                'bic': bic,
                'aic': aic,
                'log_likelihood': ll,
                'entropy': entropy,
                'silhouette': sil,
                'model': model
            })

        except Exception as e:
            warnings.warn(f"Failed for k={k}: {e}")
            continue

    if len(results) == 0:
        return {'optimal_k': None, 'results': pd.DataFrame(), 'best_model': None}

    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results])

    # Select optimal k based on criterion
    if criterion == 'bic' and 'bic' in df.columns:
        optimal_idx = df['bic'].idxmin()
    elif criterion == 'aic' and 'aic' in df.columns:
        optimal_idx = df['aic'].idxmin()
    elif criterion == 'silhouette' and 'silhouette' in df.columns:
        optimal_idx = df['silhouette'].idxmax()
    else:
        optimal_idx = 0

    optimal_k = results[optimal_idx]['k']
    best_model = results[optimal_idx]['model']

    return {
        'optimal_k': optimal_k,
        'results': df,
        'best_model': best_model
    }


def comprehensive_validation(X: np.ndarray, labels: np.ndarray,
                            clustering_func: Optional[Callable] = None,
                            external_var: Optional[np.ndarray] = None,
                            external_name: str = 'External',
                            n_bootstrap: int = 50) -> Dict:
    """
    Perform comprehensive validation of clustering results.

    Combines internal validation (metrics), stability analysis, and
    optional external validation.

    Args:
        X: Feature matrix
        labels: Cluster labels
        clustering_func: Clustering function for bootstrap (optional)
        external_var: External variable for validation (optional)
        external_name: Name of external variable
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dict with all validation results
    """
    results = {}

    # Internal validation metrics
    unique_labels = np.unique(labels[labels >= 0])

    if len(unique_labels) > 1:
        try:
            results['silhouette'] = silhouette_score(X, labels)
        except:
            results['silhouette'] = np.nan

        try:
            results['davies_bouldin'] = davies_bouldin_score(X, labels)
        except:
            results['davies_bouldin'] = np.nan
    else:
        results['silhouette'] = np.nan
        results['davies_bouldin'] = np.nan

    results['n_clusters'] = len(unique_labels)
    results['cluster_sizes'] = {k: np.sum(labels == k) for k in unique_labels}

    # Bootstrap stability
    if clustering_func is not None:
        stability = bootstrap_stability_analysis(
            X, clustering_func, n_bootstrap=n_bootstrap
        )
        results['stability'] = stability
    else:
        results['stability'] = None

    # External validation
    if external_var is not None:
        ext_val = external_validation(labels, external_var, external_name)
        results['external_validation'] = ext_val
    else:
        results['external_validation'] = None

    # Summary assessment
    results['summary'] = _generate_validation_summary(results)

    return results


def _generate_validation_summary(results: Dict) -> str:
    """Generate a textual summary of validation results."""
    parts = []

    # Cluster quality
    sil = results.get('silhouette', np.nan)
    if not np.isnan(sil):
        if sil > 0.5:
            parts.append(f"Strong cluster structure (Silhouette: {sil:.3f})")
        elif sil > 0.25:
            parts.append(f"Moderate cluster structure (Silhouette: {sil:.3f})")
        else:
            parts.append(f"Weak cluster structure (Silhouette: {sil:.3f})")

    # Stability
    stability = results.get('stability')
    if stability is not None:
        ari = stability.get('mean_ari', 0)
        grade = stability.get('stability_grade', 'Unknown')
        parts.append(f"Stability: {grade} (ARI: {ari:.3f})")

    # External validation
    ext = results.get('external_validation')
    if ext is not None:
        cramers = ext.get('cramers_v', 0)
        p_value = ext.get('p_value', 1)
        if p_value < 0.05:
            parts.append(f"Significant association with external variable (Cramer's V: {cramers:.3f})")
        else:
            parts.append("No significant association with external variable")

    return " | ".join(parts) if parts else "Validation not performed"
