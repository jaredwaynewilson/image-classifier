import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def main():

    # Get the number of unique classes. We can play with this later. 

    animals = pd.read_csv('all_animals.csv')
    animals = animals[animals.path_type=='test']
    
    embeddings = pd.read_csv('embeddings.csv')

    # Scale data
    scaling = StandardScaler()
    scaling.fit(embeddings)
    scaled_data = scaling.transform(embeddings)

    # Perform Principal Component Analysis (PCA) To reduce from 512features to 50
    principal = PCA(n_components=50)
    principal.fit(scaled_data)
    x = principal.transform(scaled_data)

    # Perform TSNE to 2 dimensions to visualize. Random state is for reproducibility
    n_dim = 2
    tsne = TSNE(n_dim, random_state=42)
    tsne_result = tsne.fit_transform(x)

    animals['tsne_0'] = tsne_result[:,0]
    animals['tsne_1'] = tsne_result[:,1]
    animals.to_csv('animals_test.csv')
                         
    # Make a little tsne df
    sns.scatterplot(data=animals, x='tsne_0', y='tsne_1', hue='true_class')
    plt.show()


if __name__ == '__main__':
    main()


