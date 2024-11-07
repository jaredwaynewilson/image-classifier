# Can hopefully remove soon because most functionality will have been moved to feature_extractor.py

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
import re
from sklearn.metrics import silhouette_samples, silhouette_score


def main():

    # Extract test files from .csv and their associated embeddings
    embeddings = pd.read_csv('embeddings.csv')
    n_samples = len(embeddings)

    # Regular expression for getting all numeric embedding values by column name. Names are 'ft_' followed by 4 digits. Matches stored in numeric_cols
    pattern = re.compile(r"ft_\d{4}")
    numeric_cols = []
    for col in embeddings.columns:
        if pattern.match(col):
            numeric_cols.append(col)
    
    # Embedding values for scaling/fitting
    numeric_embeddings = embeddings[numeric_cols]

    # Scale data
    scaling = StandardScaler()
    scaling.fit(numeric_embeddings)
    scaled_data = scaling.transform(numeric_embeddings)

    # Perform Principal Component Analysis (PCA) To reduce from X features to min(50, n_samples)
    n_comp = min(50,  n_samples)
    principal = PCA(n_components=n_comp)
    principal.fit(scaled_data)
    x = principal.transform(scaled_data)

    # Perform TSNE to 2 dimensions to visualize. Random state is for reproducibility
    n_dim = 2
    tsne = TSNE(n_dim, random_state=42)
    tsne_result = tsne.fit_transform(x)

    # Add tsne columns
    embeddings['tsne_0'] = tsne_result[:,0]
    embeddings['tsne_1'] = tsne_result[:,1]

    # Find optimal value for number of clusters
    n_min = 2
    K = list(range(n_min,10,1))
    k_silhouette_avg = np.zeros(len(K))

    # Determine best value for k with silhouette score. Run for several iterations because KMeans is stochastic
    n_iterations = 10
    for i in range(n_iterations):
        for n_clusters in K:
            # Build and fit KMeans model
            kmeanModel = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeanModel.fit_predict(numeric_embeddings)
            silhouette_avg = silhouette_score(numeric_embeddings, cluster_labels)
            k_silhouette_avg[n_clusters-n_min] += silhouette_avg

    # Divide sums by number of iterations and find max index
    k_silhouette_avg /= n_iterations
    max_index = np.argmax(k_silhouette_avg)
    optimal_n = K[max_index]

    # Build optimal Kmeans model
    kmeanModel = KMeans(n_clusters=optimal_n)
    cluster_labels = kmeanModel.fit_predict(numeric_embeddings)
    embeddings['k_label'] = cluster_labels


    # General TSNE Plot
    fig0, ax0 = plt.subplots(2, 2, figsize=(15,9))

    # TSNE scatterplot
    sns.scatterplot(data=embeddings, x='tsne_0', y='tsne_1', ax=ax0[0, 0], hue='true_class', style='true_class', palette='muted', edgecolor='black', s=50)
    # Shrink current axis by 20% and place legend to right of current axis
    box = ax0[0, 0].get_position()
    ax0[0, 0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax0[0, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Optimal number of clusters graph
    sns.lineplot(x=K, y=k_silhouette_avg, ax=ax0[0, 1])
    ax0[0,1].axvline(x=optimal_n, ls='--', color='red')
    ax0[0,1].set_xlabel('Number of Clusters')
    ax0[0,1].set_ylabel('Average Silhouette Score')


    # KMeans scatterplot
    sns.scatterplot(data=embeddings, x='tsne_0', y='tsne_1', ax=ax0[1, 0], hue='k_label', palette='deep', edgecolor='black', s=50)
    # Shrink current axis by 20% and place legend to right of current axis
    box = ax0[1, 0].get_position()
    ax0[1, 0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax0[1, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.show()


if __name__ == '__main__':
    main()
