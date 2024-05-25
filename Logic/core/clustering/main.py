import json
import numpy as np
import os
import wandb
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks

# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
ft_model = FastText()
path = '../preprocessed_documents.json'
ft_data_loader = FastTextDataLoader(path)
X, y = ft_data_loader.create_train_data_for_cluster()

ft_model.train(X)
ft_model.prepare(dataset=None, mode="save", save=True, path='./FastText_modelClustering.bin')

embeddings = []
for i in range(len(X)):
    embeddings.append(ft_model.get_query_embedding(X[i]).tolist())

with open('./embeddings.json', 'w') as f:
    json.dump(embeddings, f)

with open('./labels.json', 'w') as f:
    json.dump(y.tolist(), f)
# 1. Dimension Reduction
# TODO: Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA.
#     (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine
#     the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.
embeddings = np.array(embeddings)
dimension_Reduction = DimensionReduction()
reduced_embeddings = dimension_Reduction.pca_reduce_dimension(embeddings, n_components=2)
singular_values = dimension_Reduction.pca.singular_values_
explained_variance_ratio = dimension_Reduction.pca.explained_variance_ratio_
print(singular_values)
print(explained_variance_ratio)
# TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors
#     as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.
dimension_Reduction.wandb_plot_2d_tsne(np.array(reduced_embeddings), 'project name', 'run name')
dimension_Reduction.wandb_plot_explained_variance_by_components(np.array(reduced_embeddings), 'project name', 'run name')
#
# 2. Clustering
## K-Means Clustering
# TODO: Implement the K-means clustering algorithm from scratch.
# TODO: Create document clusters using K-Means.
# TODO: Run the algorithm with several different values of k.
# TODO: For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis
#  to choose the appropriate k.
# TODO: Plot the purity value for k using the labeled data and report the
#  purity value for the final k. (Use the provided functions in utilities)
with open('./labels.json', 'r') as f:
    clusters = json.load(f)
true_clusters = [x[0] if len(x) != 0 else 'no genre' for x in clusters]
with open('./true_clusters.json', 'w') as f:
    json.dump(true_clusters, f)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(true_clusters)

with open('./encoded_labels.json', 'w') as f:
    json.dump(encoded_labels.tolist(), f)

clustering_utils = ClusteringUtils()
k_values = [2, 3, 4, 5, 6, 7, 8, 9]
clustering_utils.plot_kmeans_cluster_scores(reduced_embeddings, encoded_labels.tolist(), k_values, 'project name', "run")
clustering_utils.visualize_elbow_method_wcss(reduced_embeddings, k_values, 'project name', "run")
clustering_utils.visualize_kmeans_clustering_wandb(np.array(reduced_embeddings), 3, 'project name', "run")
# # Hierarchical Clustering
# TODO: Perform hierarchical clustering with all different linkage methods.
# # TODO: Visualize the results.
linkage_methods = ['single', 'complete', 'average', 'ward']
project_name = 'Clustering Project'
run_name_base = 'Hierarchical Clustering'
for linkage_method in linkage_methods:
    run_name = f'{run_name_base} - {linkage_method.capitalize()} Linkage'
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(np.array(reduced_embeddings), project_name,
                                                                   linkage_method, run_name)
# # 3. Evaluation
# # TODO: Using clustering metrics, evaluate how well your clustering method is performing.
cluster_centers, cluster_labels = clustering_utils.cluster_kmeans(embeddings, 3)
clustering_metrics = ClusteringMetrics()
silhouette_score = clustering_metrics.silhouette_score(embeddings, cluster_labels)
print("k_means method:")
print(f"silhouette_score is {silhouette_score}")
purity_score = clustering_metrics.purity_score(encoded_labels, cluster_labels)
print(f"purity_score is {purity_score}")
rand_score = clustering_metrics.adjusted_rand_score(encoded_labels, cluster_labels)
print(f"rand_score is {rand_score}")

cluster_labels_single_method, _ = clustering_utils.cluster_hierarchical_single(reduced_embeddings)
print("single method:")
silhouette_score = clustering_metrics.silhouette_score(reduced_embeddings, cluster_labels_single_method)
print(f"silhouette_score is {silhouette_score}")
purity_score = clustering_metrics.purity_score(encoded_labels, cluster_labels_single_method)
print(f"purity_score is {purity_score}")
rand_score = clustering_metrics.adjusted_rand_score(encoded_labels, cluster_labels_single_method)
print(f"rand_score is {rand_score}")

cluster_labels_complete_method, _ = clustering_utils.cluster_hierarchical_complete(reduced_embeddings)
print("complete method:")
silhouette_score = clustering_metrics.silhouette_score(reduced_embeddings, cluster_labels_complete_method)
print(f"silhouette_score is {silhouette_score}")
purity_score = clustering_metrics.purity_score(encoded_labels, cluster_labels_complete_method)
print(f"purity_score is {purity_score}")
rand_score = clustering_metrics.adjusted_rand_score(encoded_labels, cluster_labels_complete_method)
print(f"rand_score is {rand_score}")

cluster_labels_average_method, _ = clustering_utils.cluster_hierarchical_average(reduced_embeddings)
print("average method:")
silhouette_score = clustering_metrics.silhouette_score(reduced_embeddings, cluster_labels_average_method)
print(f"silhouette_score is {silhouette_score}")
purity_score = clustering_metrics.purity_score(encoded_labels, cluster_labels_average_method)
print(f"purity_score is {purity_score}")
rand_score = clustering_metrics.adjusted_rand_score(encoded_labels, cluster_labels_average_method)
print(f"rand_score is {rand_score}")

cluster_labels_ward_method, _ = clustering_utils.cluster_hierarchical_ward(reduced_embeddings)
print("ward method:")
silhouette_score = clustering_metrics.silhouette_score(reduced_embeddings, cluster_labels_ward_method)
print(f"silhouette_score is {silhouette_score}")
purity_score = clustering_metrics.purity_score(encoded_labels, cluster_labels_ward_method)
print(f"purity_score is {purity_score}")
rand_score = clustering_metrics.adjusted_rand_score(encoded_labels, cluster_labels_ward_method)
print(f"rand_score is {rand_score}")