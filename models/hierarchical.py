"""Hierarchical (Agglomerative) clustering model."""

from typing import Dict, List, Any
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from models.base import BaseClusterModel


class HierarchicalModel(BaseClusterModel):
    """
    Agglomerative Hierarchical clustering algorithm.

    Builds a hierarchy of clusters by iteratively merging the closest pairs.
    Good for discovering hierarchical structure in data.
    """

    name = "Hierarchical"
    description = "Agglomerative clustering that builds cluster hierarchy bottom-up"
    supports_n_clusters = True

    def __init__(self):
        super().__init__()
        self.params = {
            'n_clusters': 3,
            'linkage': 'ward',
            'metric': 'euclidean'
        }

    def get_param_config(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'n_clusters',
                'type': 'int',
                'default': 3,
                'min': 2,
                'max': 20,
                'description': 'Number of clusters'
            },
            {
                'name': 'linkage',
                'type': 'select',
                'default': 'ward',
                'options': ['ward', 'complete', 'average', 'single'],
                'description': 'Linkage criterion'
            },
            {
                'name': 'metric',
                'type': 'select',
                'default': 'euclidean',
                'options': ['euclidean', 'manhattan', 'cosine'],
                'description': 'Distance metric (ward only supports euclidean)'
            }
        ]

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        # Ward linkage only supports euclidean metric
        if self.params['linkage'] == 'ward':
            self.params['metric'] = 'euclidean'

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        # Ward linkage requires euclidean metric
        if self.params['linkage'] == 'ward':
            self.model = AgglomerativeClustering(
                n_clusters=self.params['n_clusters'],
                linkage=self.params['linkage']
            )
        else:
            self.model = AgglomerativeClustering(
                n_clusters=self.params['n_clusters'],
                linkage=self.params['linkage'],
                metric=self.params['metric']
            )
        self.labels_ = self.model.fit_predict(X)
        return self.labels_
