import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist


def main():

    # Elbow method
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 50)

    # We want to get animal subsets
    animals = pd.read_csv('animals_test_tsne.csv')
    animal_test = animals[animals.path_type == 'test']

    squirrel_locs = np.where(animal_test.true_class == 'squirrel')[0]
    squirrels = pd.DataFrame(animals.iloc[squirrel_locs])

    # Extract embeddings of dog images
    embeddings = pd.read_csv('embeddings.csv')
    embeddings = embeddings.iloc[squirrel_locs]

    # Elbow method lists
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(embeddings)
        kmeanModel.fit(embeddings)

        distortions.append(sum(np.min(cdist(embeddings, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / embeddings.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(embeddings, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / embeddings.shape[0]
        mapping2[k] = kmeanModel.inertia_
    
    # Slope of elbow list. Once the magnitude of the slope goes below 10 percent of the max, halt
    slopes = []
    for i in range(1, len(K)):
        slope = np.abs((distortions[i] - distortions[i-1]) / (K[i] - K[i-1]))
        slopes.append(slope)

    lt = np.where(slopes <= (np.max(slopes)*.1))[0]
    stopping_point = lt[0] + 1
    print(f'stopping point: {stopping_point}')

    # Slopes plot
    plt.subplots()
    x = list(range(len(slopes)))
    plt.plot(x, slopes, 'rx-')
    plt.scatter(x[lt[0]], slopes[lt[0]], s=80, facecolors='none', edgecolors='b')
    plt.xlabel('Slope segments')
    plt.ylabel('Slope')
    plt.title('Slope comparison')

    # Elbow plot
    plt.subplots()
    plt.plot(K, distortions, 'bx-')
    plt.scatter(K[stopping_point], distortions[stopping_point], s=80, facecolors='none', edgecolors='r')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('Elbow Method using Distortion')


    # KMeans plot with optimal values
    kmeans = KMeans(n_clusters=K[stopping_point])
    kfit = kmeans.fit(embeddings)
    klabels = kmeans.fit_predict(embeddings)

    # Add predicted label column to animals df
    squirrels['predicted_label'] = ['Class_' + "{:02d}".format(label) for label in klabels]
    squirrels.sort_values(by=['predicted_label'], inplace=True)

    squirrels.to_csv('junkmates.csv')

    # Plot
    fig, ax = plt.subplots(figsize=(14,9))
    ax.set_aspect('equal', adjustable='box')
    g = sns.scatterplot(data=squirrels, x='tsne_0', y='tsne_1', style='predicted_label', hue='predicted_label', 
                    palette='viridis', edgecolor='black', legend='full')
    g.legend(loc='center right', bbox_to_anchor=(1.2, 0.5), ncol=1)

    ax.set_xlabel('tsne_1')
    ax.set_ylabel('tsne_2')
    ax.set_title('K-Means Fun')
    plt.show()
    print()


if __name__ == '__main__':
    main()