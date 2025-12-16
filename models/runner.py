"""Model runner for batch processing and comparison."""

from typing import Dict, List, Any, Optional, Type
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

from models.base import BaseClusterModel


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
    Manages the execution and comparison of multiple clustering models.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results: List[ExperimentResult] = []

    def run_single(self, model: BaseClusterModel, X: np.ndarray) -> ExperimentResult:
        """
        Run a single model configuration.

        Args:
            model: Configured model instance
            X: Scaled feature array

        Returns:
            ExperimentResult object
        """
        start_time = time.time()
        
        try:
            labels = model.fit_predict(X)
            runtime = time.time() - start_time
            metrics = model.get_metrics(X, labels)
            
            # Store readable name
            from models import MODEL_NAMES
            readable_name = MODEL_NAMES.get(model.name.lower().replace(" ", "_"), model.name)
            if 'kmeans' in model.name.lower(): readable_name = 'K-Means' # Fallback normalization
            if 'hierarchical' in model.name.lower(): readable_name = 'Hierarchical'
            if 'dbscan' in model.name.lower(): readable_name = 'DBSCAN'
            if 'gaussian' in model.name.lower() and 'bayesian' not in model.name.lower(): readable_name = 'GMM'
            if 'optics' in model.name.lower(): readable_name = 'OPTICS'
            if 'spectral' in model.name.lower(): readable_name = 'Spectral'
            if 'affinity' in model.name.lower(): readable_name = 'Affinity Propagation'
            if 'bayesian' in model.name.lower(): readable_name = 'Bayesian GMM'

            return ExperimentResult(
                model_name=readable_name,
                params=model.get_params(),
                labels=labels.copy(),
                metrics=metrics,
                runtime=runtime
            )
        except Exception as e:
            # Return failed result
            return ExperimentResult(
                model_name=model.name,
                params=model.get_params(),
                labels=np.array([]),
                metrics={'valid': False, 'error': str(e)},
                runtime=time.time() - start_time
            )

    def run_batch(self, X: np.ndarray, configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Run a batch of experiments.

        Args:
            X: Scaled feature array
            configs: List of dicts with 'model' key and param keys

        Returns:
            DataFrame of results
        """
        from models import get_model
        
        results = []
        
        # Sequential execution for stability (Streamlit + Threads can be tricky)
        for config in configs:
            model_key = config.pop('model')
            try:
                model = get_model(model_key)
                model.set_params(**config)
                result = self.run_single(model, X)
                results.append(result)
            except Exception as e:
                print(f"Error in batch run for {model_key}: {e}")
                continue

        self.results.extend(results)
        
        # Return summary DataFrame
        return pd.DataFrame([r.to_dict() for r in results])

    def get_best_model(self, metric: str = 'Silhouette') -> Optional[ExperimentResult]:
        """Get best result based on metric."""
        if not self.results:
            return None
        
        valid_results = [r for r in self.results if r.metrics.get('valid', False)]
        if not valid_results:
            return None
            
        if metric == 'Silhouette':
            return max(valid_results, key=lambda x: x.metrics.get('silhouette', -1))
        elif metric == 'Davies-Bouldin':
            return min(valid_results, key=lambda x: x.metrics.get('davies_bouldin', float('inf')))
        
        return valid_results[0]

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

        elif model == 'spectral':
            for k in k_range:
                # Add default affinity for batch runs
                configs.append({'model': 'spectral', 'n_clusters': k, 'affinity': 'rbf'})

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
        
        elif model == 'optics':
            # Default configs for OPTICS batch
            min_samples = dbscan_min_samples_range or [5, 10]
            for ms in min_samples:
                configs.append({
                    'model': 'optics',
                    'min_samples': ms,
                    'metric': 'minkowski'
                })

        elif model == 'affinity_propagation':
            # Affinity Propagation with different damping values
            for damping in [0.5, 0.7, 0.9]:
                configs.append({
                    'model': 'affinity_propagation',
                    'damping': damping
                })

        elif model == 'bayesian_gmm':
            # Bayesian GMM with different upper bounds
            for n_comp in k_range:
                configs.append({
                    'model': 'bayesian_gmm',
                    'n_components': n_comp
                })

    return configs
