"""Model runner for batch processing and comparison."""

from typing import Dict, List, Any, Optional, Type
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

from models.base import BaseClusterModel
from models.kmeans import KMeansModel
from models.hierarchical import HierarchicalModel
from models.dbscan import DBSCANModel
from models.gaussian_mixture import GaussianMixtureModel


@dataclass
class ExperimentResult:
    """Container for single experiment result."""
    model_name: str
    params: Dict[str, Any]
    labels: np.ndarray
    metrics: Dict[str, Any]
    runtime: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        return {
            'Model': self.model_name,
            'Parameters': self.params_string,
            'Clusters': self.metrics.get('n_clusters', 0),
            'Silhouette': self.metrics.get('silhouette', 0),
            'Davies-Bouldin': self.metrics.get('davies_bouldin', float('inf')),
            'Calinski-Harabasz': self.metrics.get('calinski_harabasz', 0),
            'Noise Points': self.metrics.get('n_noise', 0),
            'Runtime (s)': round(self.runtime, 3),
            'Valid': self.metrics.get('valid', False)
        }

    @property
    def params_string(self) -> str:
        """Format params for display."""
        return ", ".join(f"{k}={v}" for k, v in self.params.items())


class ModelRunner:
    """
    Run clustering experiments with batch processing support.

    Supports running multiple models with multiple parameter configurations
    and comparing results.
    """

    # Registry of available models
    MODELS: Dict[str, Type[BaseClusterModel]] = {
        'kmeans': KMeansModel,
        'hierarchical': HierarchicalModel,
        'dbscan': DBSCANModel,
        'gaussian_mixture': GaussianMixtureModel
    }

    def __init__(self):
        self.results: List[ExperimentResult] = []

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get info about all available models."""
        return {key: cls.get_info() for key, cls in self.MODELS.items()}

    def run_single(
        self,
        model_key: str,
        X: np.ndarray,
        params: Optional[Dict[str, Any]] = None
    ) -> ExperimentResult:
        """
        Run a single model with given parameters.

        Args:
            model_key: Key from MODELS registry
            X: Scaled feature array
            params: Model parameters (uses defaults if None)

        Returns:
            ExperimentResult with labels and metrics
        """
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        model = self.MODELS[model_key]()

        if params:
            model.set_params(**params)

        start_time = time.time()
        labels = model.fit_predict(X)
        runtime = time.time() - start_time

        metrics = model.get_metrics(X, labels)

        result = ExperimentResult(
            model_name=model.name,
            params=model.get_params(),
            labels=labels,
            metrics=metrics,
            runtime=runtime
        )

        return result

    def run_batch(
        self,
        X: np.ndarray,
        model_configs: List[Dict[str, Any]],
        progress_callback=None
    ) -> List[ExperimentResult]:
        """
        Run multiple experiments in batch.

        Args:
            X: Scaled feature array
            model_configs: List of dicts with 'model' key and optional params
                Example: [
                    {'model': 'kmeans', 'n_clusters': 3},
                    {'model': 'kmeans', 'n_clusters': 4},
                    {'model': 'dbscan', 'eps': 0.5}
                ]
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of ExperimentResult
        """
        self.results = []
        total = len(model_configs)

        for i, config in enumerate(model_configs):
            model_key = config.pop('model')
            params = config  # Remaining keys are params

            try:
                result = self.run_single(model_key, X, params if params else None)
                self.results.append(result)
            except Exception as e:
                # Create failed result
                self.results.append(ExperimentResult(
                    model_name=model_key,
                    params=params,
                    labels=np.array([]),
                    metrics={'valid': False, 'error': str(e)},
                    runtime=0
                ))

            if progress_callback:
                progress_callback(i + 1, total)

        return self.results

    def run_all_models_with_k_range(
        self,
        X: np.ndarray,
        k_range: range = range(2, 11),
        include_dbscan: bool = True,
        dbscan_eps_values: List[float] = [0.3, 0.5, 1.0, 2.0, 3.0],
        progress_callback=None
    ) -> List[ExperimentResult]:
        """
        Run all models with a range of cluster values.

        Args:
            X: Scaled feature array
            k_range: Range of k values for models that support n_clusters
            include_dbscan: Whether to include DBSCAN experiments
            dbscan_eps_values: Epsilon values to try for DBSCAN
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of ExperimentResult
        """
        configs = []

        # K-Means with different k values
        for k in k_range:
            configs.append({'model': 'kmeans', 'n_clusters': k})

        # Hierarchical with different k and linkage
        for k in k_range:
            for linkage in ['ward', 'complete', 'average']:
                configs.append({
                    'model': 'hierarchical',
                    'n_clusters': k,
                    'linkage': linkage
                })

        # Gaussian Mixture with different k and covariance types
        for k in k_range:
            for cov_type in ['full', 'tied', 'diag']:
                configs.append({
                    'model': 'gaussian_mixture',
                    'n_components': k,
                    'covariance_type': cov_type
                })

        # DBSCAN with different eps values
        if include_dbscan:
            for eps in dbscan_eps_values:
                for min_samples in [3, 5, 10]:
                    configs.append({
                        'model': 'dbscan',
                        'eps': eps,
                        'min_samples': min_samples
                    })

        return self.run_batch(X, configs, progress_callback)

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()

        data = [r.to_dict() for r in self.results]
        df = pd.DataFrame(data)

        # Sort by silhouette score (descending)
        df = df.sort_values('Silhouette', ascending=False)

        return df

    def get_best_result(self, metric: str = 'silhouette') -> Optional[ExperimentResult]:
        """
        Get best result by specified metric.

        Args:
            metric: 'silhouette', 'davies_bouldin', or 'calinski_harabasz'

        Returns:
            Best ExperimentResult or None
        """
        valid_results = [r for r in self.results if r.metrics.get('valid', False)]
        if not valid_results:
            return None

        if metric == 'silhouette':
            return max(valid_results, key=lambda r: r.metrics.get('silhouette', 0))
        elif metric == 'davies_bouldin':
            return min(valid_results, key=lambda r: r.metrics.get('davies_bouldin', float('inf')))
        elif metric == 'calinski_harabasz':
            return max(valid_results, key=lambda r: r.metrics.get('calinski_harabasz', 0))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_top_n_results(self, n: int = 5, metric: str = 'silhouette') -> List[ExperimentResult]:
        """Get top N results by specified metric."""
        valid_results = [r for r in self.results if r.metrics.get('valid', False)]

        if metric == 'davies_bouldin':
            sorted_results = sorted(
                valid_results,
                key=lambda r: r.metrics.get('davies_bouldin', float('inf'))
            )
        else:
            key_func = lambda r: r.metrics.get(metric, 0)
            sorted_results = sorted(valid_results, key=key_func, reverse=True)

        return sorted_results[:n]

    def clear_results(self):
        """Clear all stored results."""
        self.results = []


def create_experiment_configs(
    models: List[str],
    k_range: range,
    dbscan_eps_range: List[float] = None,
    dbscan_min_samples_range: List[int] = None
) -> List[Dict[str, Any]]:
    """
    Helper to create experiment configurations.

    Args:
        models: List of model keys ('kmeans', 'hierarchical', etc.)
        k_range: Range of k values
        dbscan_eps_range: Epsilon values for DBSCAN
        dbscan_min_samples_range: Min samples values for DBSCAN

    Returns:
        List of config dicts for run_batch
    """
    configs = []

    for model in models:
        if model == 'kmeans':
            for k in k_range:
                configs.append({'model': 'kmeans', 'n_clusters': k})

        elif model == 'hierarchical':
            for k in k_range:
                configs.append({'model': 'hierarchical', 'n_clusters': k})

        elif model == 'gaussian_mixture':
            for k in k_range:
                configs.append({'model': 'gaussian_mixture', 'n_components': k})

        elif model == 'dbscan':
            eps_values = dbscan_eps_range or [0.5, 1.0, 2.0]
            min_samples = dbscan_min_samples_range or [5]
            for eps in eps_values:
                for ms in min_samples:
                    configs.append({
                        'model': 'dbscan',
                        'eps': eps,
                        'min_samples': ms
                    })

    return configs
