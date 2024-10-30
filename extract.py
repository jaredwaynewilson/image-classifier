import pandas as pd
import numpy as np
from torchvision import transforms, datasets, models
import torch
from PIL import Image


# The recommended mean parameters for transforms with ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
# The recommended standard deviation parameters for transforms with ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225]
# determines the height and width of a transformed image.
RESIZE_SHAPE = (224, 224)


def get_embeddings(df, model, in_ftrs, desired_features):

    # Alter fully connected layer
    model.fc = torch.nn.Linear(in_ftrs, desired_features)
    df_copy = df.copy(deep=True)

    # Get image paths of test images
    val_rows = np.where(df_copy.path_type == 'test')[0]
    val_paths = df_copy.iloc[val_rows].image_path.values
    
    # Create embeddings array
    nrows = val_paths.shape[0]
    embeddings = np.empty((nrows, desired_features))

    # Set model to evaluation mode. Define transforms
    model.eval()

    # Some image reshaping will be used
    val_transform = transforms.Compose([
            transforms.Resize(RESIZE_SHAPE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    
    # Iterate through paths and extract embeddings
    for idx, entry in enumerate(val_paths):

        # transform img
        my_img = Image.open(entry)
        my_img = val_transform(my_img)
        batch_t = torch.unsqueeze(my_img, dim=0)

        # Infer 
        out = model(batch_t)
        out = out.detach().numpy().flatten()

        # fill embeddings array
        embeddings[idx] = out
    
    # Convert embeddings to df
    embeddings = pd.DataFrame(embeddings)
    
    return embeddings


def main():
    filename = 'all_animals.csv'
    df = pd.read_csv(filename)
    n_classes = df.true_class.nunique()

    # Import model weights and model
    model = models.resnet18()
    model_params = 'best_model_params.pt'

    # Alter model fc and load params
    in_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(in_ftrs, n_classes)
    model.load_state_dict(torch.load(model_params))

    # Outfeatures asks how many unique features we want our CNN to recognize. 
    # It affects the number of columns in our embeddings .csv
    out_features = 512

    # Get embeddings
    embeddings = get_embeddings(df, model, in_ftrs, out_features)
    embeddings.to_csv('embeddings.csv',index=False)


if __name__ == '__main__':
    main()
