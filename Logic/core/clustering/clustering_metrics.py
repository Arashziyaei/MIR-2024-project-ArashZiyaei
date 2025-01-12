import numpy as np

from typing import List

import sklearn.metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix


class ClusteringMetrics:

    def __init__(self):
        pass

    def silhouette_score(self, embeddings: List, cluster_labels: List) -> float:
        """
        Calculate the average silhouette score for the given cluster assignments.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points.
        cluster_labels: List
            A list of cluster assignments for each data point.

        Returns
        --------
        float
            The average silhouette score, ranging from -1 to 1, where a higher value indicates better clustering.
        """
        return silhouette_score(embeddings, cluster_labels)

    def purity_score(self, true_labels: List, cluster_labels: List) -> float:
        """
        Calculate the purity score for the given cluster assignments and ground truth labels.

        Parameters
        -----------
        true_labels: List
            A list of ground truth labels for each data point (Genres).
        cluster_labels: List
            A list of cluster assignments for each data point.

        Returns
        --------
        float
            The purity score, ranging from 0 to 1, where a higher value indicates better clustering.
        """
        size_clust = np.max(cluster_labels) + 1
        len_clust = len(cluster_labels)
        c_labels = [None] * size_clust
        for i in range(len_clust):
            index = cluster_labels[i]
            if c_labels[index] is None:
                c_labels[index] = true_labels[i]
            else:
                c_labels[index] = np.hstack((c_labels[index], true_labels[i]))

        purity = 0
        for c in c_labels:
            y = np.bincount(c)
            maximum = np.max(y)
            purity += maximum

        purity = purity / len_clust

        return purity

    def adjusted_rand_score(self, true_labels: List, cluster_labels: List) -> float:
        """
        Calculate the adjusted Rand index for the given cluster assignments and ground truth labels.

        Parameters
        -----------
        true_labels: List
            A list of ground truth labels for each data point (Genres).
        cluster_labels: List
            A list of cluster assignments for each data point.

        Returns
        --------
        float
            The adjusted Rand index, ranging from -1 to 1, where a higher value indicates better clustering.
        """
        return adjusted_rand_score(true_labels, cluster_labels)
