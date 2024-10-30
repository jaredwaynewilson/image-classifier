# Imports



from torchvision import transforms, datasets, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision.models import ResNet18_Weights

# Constants
# The recommended mean parameters for transforms with ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]

# The recommended standard deviation parameters for transforms with ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225]

# determines the height and width of a transformed image.
RESIZE_SHAPE = (224, 224)

# Global variables
# Directory containing training and validation images
data_dir = 'TransformedBatch/'

# Holds model weights as they update
model_params = 'best_model_params.pt'

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(RESIZE_SHAPE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
    'val': transforms.Compose([
        transforms.Resize(RESIZE_SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]),
}

# This loads datasets into the CNN
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'val']}

# access gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Gets dataset sizes and class names. These won't change throughout this script
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_len =  len(image_datasets['train'].classes)

# Training values
dropout_rate = 0.5
learning_rate = 1e-5
weight_decay = 5e-5


def remove_ds(some_list):
    """ If the ds file from a mac exists, remove it from the os directory list"""
    my_copy = some_list.copy()

    # Delete the macOS .DS_Store if it exists
    if ".DS_Store" in my_copy:
        id = my_copy.index(".DS_Store")
        del my_copy[id]

    return my_copy


def train_model(model, criterion, optimizer, scheduler, num_epochs=12):
    since = time.time()

    # Let's create a few lists so that we can plot this
    epochs = list(range(num_epochs))
    val_loss, train_loss, val_err, train_err = [[None] * len(epochs) for i in range(4)]

    # # Live plotting
    f1, ax1 = plt.subplots(figsize=(14,9))
    ymax_flag = True
    ymax_loss = 0
    ymax_err = 0
    
    # Create a torch file to save training checkpoints
    with open(model_params, 'w'):

        torch.save(model.state_dict(), model_params)
        best_err = 100.00

        for epoch in range(num_epochs):
            epoch_time = time.time()
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_err = 100 - 100 *(running_corrects.double() / dataset_sizes[phase])


                if phase == "train":
                    train_loss[epoch] = epoch_loss
                    train_err[epoch] = float(epoch_err)
                    if ymax_flag:
                        ymax_loss = epoch_loss
                        ymax_err = float(epoch_err)
                        ymax_flag = False
                elif phase == "val":
                    val_loss[epoch] = epoch_loss
                    val_err[epoch] = float(epoch_err)

                print(f'{phase} Loss: {epoch_loss:.4f} | {phase} Error: {epoch_err:.4f} %')

                # deep copy the model
                if phase == 'val' and epoch_err < best_err:
                    best_err = epoch_err
                    torch.save(model.state_dict(), model_params)
                    # in this script
            ep_time = time.time() - epoch_time
            print(f'Epoch {epoch} completed in {ep_time // 60:.0f}m {ep_time % 60:.0f}s')

        ax1.grid(visible=True)
        ax1.set_axisbelow(True)

        line1, = ax1.plot(epochs, train_loss, c='steelblue', label='train')
        line2, = ax1.plot(epochs, val_loss, c='firebrick', label='val')


        ax1.legend(handles=[line1, line2], fontsize=16)
                  
        ax1.set_xlim(0, num_epochs)
        ax1.set_ylim(bottom=0, top=ymax_loss)
        ax1.set_ylabel("Loss", fontsize=14)
        ax1.set_xlabel("Epoch", fontsize=14)

        
        plt.title("ResNet-18 Loss Trajectory on 58k Image Set", fontsize=18)
        
        f1.savefig('loss_trajectory_3class58k.jpg', dpi=300, bbox_inches='tight')


        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Error: {best_err:4f}')

        # load best model weights
        model.load_state_dict(torch.load(model_params))

    return model


def main():
    # Import resnet model
    model_ft = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Get number of features; this will be outputted
    num_ftrs = model_ft.fc.in_features

    # Change fully connected layer and create loss/optimization functions
    model_ft.fc = nn.Linear(num_ftrs, class_len)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate decay by a factor of 0 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0)

    # Train it on image data
    j_model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=12)


if __name__ == '__main__':
    main()