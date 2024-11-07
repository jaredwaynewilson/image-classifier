# Python3 program with ...

# Make sure to test this

import pandas as pd
import numpy as np
from torchvision import transforms, models
import torch
from PIL import Image
import torch.nn as nn


# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406] # The recommended mean parameters for transforms with ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225] # The recommended standard deviation parameters for transforms with ImageNet
RESIZE_SHAPE = (224, 224) # determines the height and width of a transformed image.


class FeatureExtractor():

    def __init__(self, df, model):
        self.df = df
        self.model = model


    def get_embeddings(self, transform, num_ftrs):
        """ The thing that it does

        Args:
            - transform {
                type:
                description:
            }
            - output_features {
                type:
                description:
            }
        
        Returns:
            - embeddings {
                type: np.array of shape (len(paths), output_features)
                description: contains the features (embeddings) for the machine learning model's evaluation of each image. Each row contains
                an array of size len(paths) with each value in the array representing a feature value. 
            }
        """
        
        # Generate empty embeddings array
        nrows = self.df.shape[0]
        embeddings = np.empty((nrows, num_ftrs))

        # Set model to evaluation mode. Define transforms
        self.model.eval()
        
        # Iterate through paths and extract embeddings
        for idx, entry in enumerate(self.df.image):

            # transform image
            my_img = Image.open(entry).convert('RGB')
            my_img = transform(my_img)
            batch_t = torch.unsqueeze(my_img, dim=0)

            # Make inference and extract features
            out = self.model(batch_t)
            out = out.detach().numpy().flatten()

            # Fill embeddings array
            embeddings[idx] = out
    
        return embeddings
    

def main():

    # Paths file
    filename = 'evaluation.csv'
    df = pd.read_csv(filename)

    n_classes = df.true_class.nunique()
    print(f'number of unique classes: {n_classes}')

    # Import model, store original number of features, adjust model fc to load weights, and load weights
    model = models.resnet18()
    model_params = 'best_model_params_squirrel.pt'
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    model.load_state_dict(torch.load(model_params))
    model.fc = nn.Linear(num_ftrs, num_ftrs)


    # Image transform for regularization
    transform = transforms.Compose([
                transforms.Resize(RESIZE_SHAPE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])


    # Instantiate object
    feature_extractor = FeatureExtractor(df, model)

    # Extract embeddings
    embeddings = feature_extractor.get_embeddings(transform=transform, num_ftrs=num_ftrs)

    # Convert embeddings to a dataframe and write to .csv
    col_names = ['ft_' + ('%04d' % i) for i in range(num_ftrs)]
    embeddings = pd.DataFrame(embeddings, columns=col_names)

    df = df.reset_index(drop=True)
    test_embeddings = pd.concat([df, embeddings], axis=1)
    test_embeddings.to_csv('embeddings.csv',index=False)


if __name__ == '__main__':
    main()
