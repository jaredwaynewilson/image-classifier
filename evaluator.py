# Imports
from torchvision import transforms, models
import torch
import torch.nn as nn
import os
from PIL import Image
from scipy.special import softmax
import pandas as pd


 # Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406] # The recommended mean parameters for transforms with ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225] # The recommended standard deviation parameters for transforms with ImageNet
RESIZE_SHAPE = (224, 224) # Height and width of a transformed image.


def remove_ds(some_list):
    """ If the ds file from a mac exists, remove it from the os directory list
    
    Args: 
        - some_list {
            type: list
            description: A list generated from the function os.listdir
        }

    Returns: None
    """
    my_copy = some_list.copy()

    # Delete the macOS .DS_Store if it exists
    if ".DS_Store" in my_copy:
        id = my_copy.index(".DS_Store")
        del my_copy[id]

    return my_copy


def main():
    # Import resnet model
    model = models.resnet18()

    # Get number of input features for the model's fully connectedt layer. 
    num_ftrs = model.fc.in_features

    # Pull directory
    pull_dir = 'parsed_squirrel/'
    model_params = 'best_model_params_squirrel.pt'

    # Class length is the number of class folders in the training subdirectory
    train_sub = pull_dir + 'train'
    class_dirs = os.listdir(train_sub)
    class_dirs = remove_ds(class_dirs)
    class_len = len(class_dirs)

    # Change fully connected layer and load optimized model weights
    model.fc = nn.Linear(num_ftrs, class_len)
    model.load_state_dict(torch.load(model_params))
    model.eval()

    # Data augmentation and normalization for training
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(RESIZE_SHAPE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    }

    # Test model
    test_dir = pull_dir + 'test/'
    classes = os.listdir(test_dir)

    correct_prediction = 0
    incorrect_prediction = 0

    labels = os.listdir(test_dir)
    labels.sort()

    predicted_labels = []
    true_labels = []
    percent_leads = []
    correctness = []
    max_probs = []
    image_paths = []

    for true_class in classes:
        folder_path = test_dir + true_class + '/'
        images = os.listdir(folder_path)
        images = remove_ds(images)

        for image in images:
            im_path = folder_path + image
            image_paths.append(im_path)
            my_img = Image.open(im_path).convert('RGB')
            my_img = data_transforms['test'](my_img)
            batch_t = torch.unsqueeze(my_img, dim=0)
            output = model(batch_t)
            output = output.detach().numpy()

            probabilities = list(softmax(output).flatten())

            max_prob = max(probabilities)
            max_probs.append(max_prob)
            max_idx = probabilities.index(max_prob)
            pred_label = labels[max_idx]
            predicted_labels.append(pred_label)
            true_labels.append(true_class)

            probabilities.pop(max_idx)
            percent_lead = max_prob - max(probabilities)
            percent_leads.append(percent_lead)
            conf_str = '{:.4%}'.format(percent_lead)

            if true_class == pred_label:
                correct_prediction += 1
                correctness.append('correct')
            else:
                incorrect_prediction += 1
                correctness.append('incorrect')

    df = pd.DataFrame({'image': image_paths, 'true_class': true_labels, 'predicted_class': predicted_labels, 'correctness': correctness,'max_probability':max_probs, 'percent_lead': percent_leads})
    df.to_csv('evaluation.csv', index=False)
    print(df.head())

    print()
    print('Test Set Evaluation')
    print('-'*30)
    print(f'Correct preditions: {correct_prediction}')
    print(f'Inrrect preditions: {incorrect_prediction}')
    percent_correct = '{:.4%}'.format(correct_prediction/float(correct_prediction + incorrect_prediction))
    print(f'Percent correct: {percent_correct}')

    # Write text file with some evaluation info
    file0 = open("evaluator_data.txt", "w")  # write mode
    file0.write(f'Test Set Evaluation\n')
    file0.write(f'Images tested: {(correct_prediction+incorrect_prediction)}\n')
    file0.write(f'Correct predictions: {correct_prediction}\n')
    file0.write(f'Incorrect predictions: {incorrect_prediction}\n') 
    file0.write(f'Percent correct: {percent_correct}\n')

    # fig, ax = plt.subplots()
    # sns.histplot(data=df, x="max_probability", hue="correctness", multiple="stack", bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1.0])
    # ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1.0])
    # plt.show()



if __name__ == '__main__':
    main()