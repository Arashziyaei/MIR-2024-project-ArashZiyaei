import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import Counter
from .clustering_metrics import *
from .dimension_reduction import DimensionReduction


class ClusteringUtils:

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        X = np.array(emb_vecs)
        centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
        for _ in range(max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        return centroids.tolist(), labels.tolist()

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        word_counts = Counter()
        for doc in documents:
            words = doc.split()
            word_counts.update(words)
        return word_counts.most_common(top_n)

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum
        of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each
        input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances
        between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are
        closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the
        closest centroid until convergence or a maximum number of iterations is reached. This function uses a random
        initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        cluster_centers, cluster_labels = self.cluster_kmeans(emb_vecs, n_clusters)

        # Calculate WCSS
        wcss = 0.0
        for i, vec in enumerate(emb_vecs):
            center = cluster_centers[cluster_labels[i]]
            wcss += np.sum((np.array(vec) - np.array(center)) ** 2)
        return cluster_centers, cluster_labels, wcss

    def cluster_hierarchical_single(self, emb_vecs: List):
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage='single')
        cluster_labels = clustering.fit_predict(emb_vecs)
        return cluster_labels, linkage(np.array(emb_vecs), method='single')

    def cluster_hierarchical_complete(self, emb_vecs: List):
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage='complete')
        cluster_labels = clustering.fit_predict(emb_vecs)
        return cluster_labels, linkage(np.array(emb_vecs), method='complete')

    def cluster_hierarchical_average(self, emb_vecs: List):
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage='average')
        cluster_labels = clustering.fit_predict(emb_vecs)
        return cluster_labels, linkage(np.array(emb_vecs), method='average')

    def cluster_hierarchical_ward(self, emb_vecs: List):
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        clustering = AgglomerativeClustering(linkage='ward')
        cluster_labels = clustering.fit_predict(emb_vecs)
        return cluster_labels, linkage(np.array(emb_vecs), method='ward')

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by
        logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data
        point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # Initialize wandb
        wandb.init(project=project_name, name=run_name)

        # Perform K-means clustering
        # TODO
        cluster_centers, cluster_labels = self.cluster_kmeans(data.tolist(), n_clusters=n_clusters)

        # Plot the clusters
        # TODO
        data_2d = DimensionReduction().convert_to_2d_tsne(data)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter([x[0] for x in data_2d], [x[1] for x in data_2d], c=cluster_labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'Kmeans Clustering with {n_clusters} Clusters')
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
        plt.show()
        # Log the plot to wandb
        # TODO
        wandb.log({"Kmeans Clustering": wandb.Image(plt.gcf())})

        # Close the plot display window if needed (optional)
        # TODO
        plt.close()
        wandb.finish()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot,
        which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the
        data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering
        process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward",
            "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        wandb.init(project=project_name, name=run_name)
        # Perform hierarchical clustering
        # TODO
        if linkage_method == 'ward':
            _, cluster_labels = self.cluster_hierarchical_ward(data.tolist())
        elif linkage_method == 'average':
            _, cluster_labels = self.cluster_hierarchical_average(data.tolist())
        elif linkage_method == 'complete':
            _, cluster_labels = self.cluster_hierarchical_complete(data.tolist())
        else:
            _, cluster_labels = self.cluster_hierarchical_single(data.tolist())

        # Create linkage matrix for dendrogram
        # TODO
        plt.figure(figsize=(10, 8))
        dendrogram(np.array(cluster_labels))
        plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()

        wandb.log({"Clustering dendrogram": wandb.Image(plt.gcf())})
        plt.close()
        wandb.finish()

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and
        silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        silhouette_scores = []
        purity_scores = []
        # Calculating Silhouette Scores and Purity Scores for different values of k
        for k in k_values:
            # TODO
            cluster_centers, cluster_labels = self.cluster_kmeans(embeddings, k)
            clustering_metrics = ClusteringMetrics()
            silhouette_score = clustering_metrics.silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(silhouette_score)
            purity_score = clustering_metrics.purity_score(true_labels, cluster_labels)
            purity_scores.append(purity_score)
            # and visualize it.
            # TODO

        # Plotting the scores
        # TODO
        wandb.init(project=project_name, name=run_name)
        plt.figure(figsize=(14, 7))
        plt.plot(k_values, silhouette_scores, label='Silhouette Score', marker='o')
        plt.plot(k_values, purity_scores, label='Purity Score', marker='o')
        plt.title('Cluster Scores for different values of K')
        plt.xlabel('Number of clusters')
        plt.ylabel('score')
        plt.legend()
        plt.grid(True)
        plt.show()

        wandb.log({"Cluster Scores": wandb.Image(plt.gcf())})
        plt.close()
        wandb.finish()

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering
        based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering.
        It involves plotting the WCSS values for different values of K (number of clusters) and finding
        the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is
        considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting
        WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        wandb.init(project=project_name, name=run_name)

        # Compute WCSS values for different K values
        wcss_values = []
        for k in k_values:
            # TODO
            wcss_values.append(self.cluster_kmeans_WCSS(embeddings, k)[2])
        # Plot the elbow method
        # TODO
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, wcss_values, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('WCSS')
        plt.grid(True)
        plt.show()

        # Log the plot to wandb
        wandb.log({"Elbow Method": wandb.Image(plt.gcf())})

        plt.close()
        wandb.finish()