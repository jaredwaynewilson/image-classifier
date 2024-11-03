# Python3 program that utilizes an image file path dataset generated from 'file_parser.py' to train a machine learning model 
# imported from PyTorch. The object accomplishing this task is known as a ModelTrainer. The ModelTrainer's model weights begin
# as those from the ImageNet1k set, but these weights are updated at the end of each training epoch as the model is trained on
# the specified datase. These weights are stored locally via 'best_model_params.pt'. ModelTrainer displpays training accuracy, 
# training loss, validation accuracy, and validation loss for each epoch. 


# Imports
from torchvision import transforms, datasets, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import matplotlib.pyplot as plt
import time
from torchvision.models import ResNet18_Weights


# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406] # The recommended mean parameters for transforms with ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225] # The recommended standard deviation parameters for transforms with ImageNet
RESIZE_SHAPE = (224, 224) # Height and width of a transformed image.


class ModelTrainer:

    def __init__(self, pull_dir, model_params, data_transforms, batch_size=32, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.pull_dir = pull_dir # Directory containing training and validation images
        self.batch_size = batch_size # Batch size for dataloader
        self.model_params = model_params # Holds model weights as they update. 
        self.data_transforms = data_transforms

        # This loads datasets into the CNN
        # Dataloader code is extracted from the pytorch tutorial at https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.pull_dir, x), data_transforms[x]) for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
        
        # Program runs on Cuda GPU if available; otherwise, runs on local CPU
        self.device = device

        # Finds number of images in training dataset and validation dataset
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}

        # Number of classes the model is trained on. Determined by the number of subdirectories in train
        self.class_len =  len(self.image_datasets['train'].classes)


    def train(self, model, criterion, optimizer, scheduler, num_epochs=12):
        """ Performs machine learning model training according to arguments specified in the function. Outputs training and validation 
            data at the conclusion of each epoch and updates a loss trajectory plot that is saved to a .jpg file upon completion of training. 

        Args: 
            - model {
                type: torchvision.model
                description: classification model available via torchvision's set of models. This is the model that will be training on 
                the data.
            }
            - criterion {
                type: torch.nn
                description: a standard on which the error is measured between elements in an input x and output y.
            }
            - optimizer {
                type: torch.optim.Optimizer
                description: adjusts the model's parameters to minimize los and error during training.
            }
            - scheduler {
                type: torch.optim.lr_scheduler
                decsription: Decays the learning rate of each parameter group by gamma every step_size epochs.
            }
            - num_epochs {
                type: int
                description: an epoch occurs after every data sample in a training set is used to update a model's parameters. The number of
                epochs determines how many cycles of training occur. 
            }
        """
        since = time.time()

        # Let's create a few lists so that we can plot this
        epochs = list(range(num_epochs))
        val_loss, train_loss, val_err, train_err = [[None] * len(epochs) for _ in range(4)]

        # # Live plotting
        f1, ax1 = plt.subplots(figsize=(14,9))
        ymax_flag = True
        ymax_loss = 0
        
        # Create a torch file to save training checkpoints
        with open(self.model_params, 'w'):

            torch.save(model.state_dict(), self.model_params)
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
                    for inputs, labels in self.dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

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

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_err = 100 - 100 *(running_corrects.double() / self.dataset_sizes[phase])


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
                        torch.save(model.state_dict(), self.model_params)
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

            
            plt.title("ResNet-18 Loss Trajectory on Image Set", fontsize=18)
            f1.savefig('loss_trajectory.jpg', dpi=300, bbox_inches='tight')


            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Error: {best_err:4f}')

            # load best model weights
            model.load_state_dict(torch.load(self.model_params))

        return model


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
    model_ft = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Get number of input features for the model's fully connectedt layer. 
    num_ftrs = model_ft.fc.in_features

    # Pull directory
    pull_dir = 'parsed_hymenoptera/'
    model_params = 'best_model_params.pt'

    # Class length is the number of class folders in the training subdirectory
    train_sub = pull_dir + 'train'
    class_dirs = os.listdir(train_sub)
    class_dirs = remove_ds(class_dirs)
    class_len = len(class_dirs)

    # Change fully connected layer and create loss/optimization functions
    model_ft.fc = nn.Linear(num_ftrs, class_len)

    # Computing device
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Connect model to device and define criterion, optimizer, and learning rate scheduler
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

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

    # Define  hyperparameters
    learning_rate = 1.5e-5
    weight_decay = 5e-5
    n_epochs = 14
    step_size = 3
    gamma = 0.4

    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=learning_rate, weight_decay=weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    # instantiate object
    model_trainer = ModelTrainer(pull_dir, model_params, data_transforms)

    # Train model
    model_trainer.train(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=n_epochs)


if __name__ == '__main__':
    main()
