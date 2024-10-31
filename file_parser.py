# Python3 program that generates a file parser object known as FileParser. FileParser is used to split a directory of images
# into training, validation, and testing sets for machine learning. Though machine learning libaries have similar functions, this 
# class is designed with image sets in mind that have a main directory holding all images and a set of subdirectories with each 
# subdirectory containing only images of a specific class. This class is denoted by the name of the subdirectory. 
#
# Example:
#       Fish
#           |-- one_fish 
#           |          |-- goldfish.jpg
#           |          |-- oldfish.jpg
#           |
#           |-- two_fish
#           |          |-- fishes.jpg
#           |          |-- sharks.jpg
#           |
#           |-- red_fish
#           |          |-- sockeye_salmon.jpg
#           |          |-- red_snapper.jpg
#           |
#           |-- blue_fish
#                      |-- tailor.jpg
#                      |-- bluefish.jpg
#
#   FileParser also has a function that generates a .csv file containing information regarding where each image in the original directory is
#   sent
#
#   Last updated: 10-31-2024
#   JW

import time
import os
import math
from PIL import Image
import pandas as pd
import numpy as np


class FileParser():

    def __init__(self, pull_path, push_path):
        self.pull_path = pull_path # Directory storing images prior to parsing
        self.push_path = push_path # Directory that will store parsed subdirectories for training, validation, and testing
        self.path_type = [] # train, val, or test
        self.true_class = [] # true class type, i.e. the name of the  directory containing the image
        self.image_paths = [] # image titles
        self.embeddings = [] # embeddings
        self.n_images = 0


    def remove_ds(self, some_list):
        """ If the .DS file from a macos system is present in a directory list, removes the .DS file.
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
            my_id = my_copy.index(".DS_Store")
            del my_copy[my_id]

        return my_copy


    def parse(self, training_weight=0.7, validation_weight=0.2, test_weight=0.1):
        """ If the .DS file from a macos system is present in a directory list, removes the .DS file.

        Args: 
            - training_weight {
                type: float
                description: The proportion of images allocated to the training directory. Default is 70 percent, or 0.7.
                             This set is used to train a machine learning algorithm.
            }
            - validation_weight {
                type: float
                description: The proportion of images allocated to the validation directory. Default is 20 percent, or 0.1.
                             This set is used to validate a machine learning algorithm's accuracy as it is learning.
            }
            - test_weight {
                type: float
                description: The proportion of images allocated to the training directory. Default is 70 percent, or 0.7.
                             This set is used to evaluate a machine learning algorithm's accuracy with real-world data. 
            }
        
        Returns: None
        """

        # Dictionary created for ease of writing boundary tests and associated errors
        weight_dict = {'training_weight':training_weight, 'validation_weight':validation_weight, 'test_weight':test_weight}

        # Boundary test: Raises a value error if the sum of training, validation, and test weights exceeds 1.0
        if (round((training_weight + validation_weight + test_weight), 5) != 1.):
            raise ValueError(f'The sum of training, validation, and test weights must be equal to 1.0! | Value: {training_weight + validation_weight + test_weight}')
        
        # Boundary test: Raises a value error if any of training_weight, validation_weight, or test_weight is not in the range of 0.0 <= weight <= 1.0
        for key in weight_dict:
            if ((weight_dict[key] < 0.0) or (weight_dict[key] > 1.0)):
                raise ValueError(f'{key} out of range! Value: {weight_dict[key]} | Value: {weight_dict[key]}')
                   
        # Get directory and remove .DS file from directory list if present.
        parent_dir = self.remove_ds(os.listdir(self.pull_path))

        # Subdirectory paths
        paths = ['train', 'val', 'test']

        # Generate subdirectories for each class type
        for path in paths:
            # Make class directories
            for child_dir in parent_dir:
                class_dir = self.push_path + path + "/" + child_dir
                if not os.path.exists(class_dir):
                    os.mkdir(class_dir)

        # Count the number of images
        for child in parent_dir:
            child_path = self.pull_path + child + '/'
            child_dir = os.listdir(child_path)
            child_dir = self.remove_ds(child_dir)
            for image in child_dir:
                self.n_images += 1

        print(f"Total number of Images: {self.n_images}")
        print('-'*60)
        successful_filereads = self.n_images
        species_left = len(parent_dir)

        
        # Parse each subdirectory according to the weights
        global_time = time.time()
        for child in parent_dir:
            start_time = time.time()
            child_path = self.pull_path + child + '/'
            child_dir = os.listdir(child_path)
            child_dir = self.remove_ds(child_dir)

            child_dir_nfiles = len(child_dir)

            train_val_split_idx = int(math.floor(child_dir_nfiles * training_weight))
            val_test_split_idx = train_val_split_idx + int(math.floor(child_dir_nfiles * validation_weight))
            
            # Append to true class
            for image in child_dir:
                self.true_class.append(child)

            # Split for training and validation
            train_images = child_dir[:train_val_split_idx]
            validation_images = child_dir[train_val_split_idx:val_test_split_idx]
            test_images = child_dir[val_test_split_idx:]
            
            # Training
            for image in train_images:
                
                self.path_type.append('train')
                img_path = child_path + image
                try:
                    my_img = Image.open(child_path + image)
                    self.image_paths.append(img_path)
                except TimeoutError:
                    # If image does not cooperate after a fair amount of time, skip
                    successful_filereads -= 1
                    continue

                # Once finished, save image to a path
                path = self.push_path + 'train/' + child + '/' + image
                my_img.save(path)

            # Validation
            for image in validation_images:
                self.path_type.append('val')
                img_path = child_path + image
                try:
                    my_img = Image.open(img_path)
                    self.image_paths.append(img_path)
                except TimeoutError:
                    # If image does not cooperate after a fair amount of time, skip
                    successful_filereads -= 1
                    continue
                
                path = self.push_path + 'val/' + child + '/' + image
                my_img.save(path)
            
            # Testing
            for image in test_images:
                self.path_type.append('test')
                img_path = child_path + image
                try:
                    my_img = Image.open(img_path)
                    self.image_paths.append(img_path)
                except TimeoutError:
                    # If image does not cooperate after a fair amount of time, skip
                    successful_filereads -= 1
                    continue
                
                path = self.push_path + 'test/' + child + '/' + image
                my_img.save(path)

            print(f"100 Percent Finished parsing directory '{child}'")
            time_elapsed = time.time() - start_time
            print("Elapsed time: " + time.strftime("%H:%M:%S.{}".format(str(time_elapsed % 1)[2:])[:15], time.gmtime(time_elapsed)))
            print(f"Folder '{child}' Completed.")

            species_left -= 1
            print(f"Folders Left to Process: {species_left}")
            print('-' * 60)

        print("Finished with All")
        print(f"Successful file reads: {successful_filereads}")
        total_time_elapsed = time.time() - global_time
        print("Total Elapsed Time: " + time.strftime("%H:%M:%S.{}".format(str(time_elapsed % 1)[2:])[:15],
                                            time.gmtime(total_time_elapsed)))
        
    
    def write_csv(self, name, val=True):
        """ Converts the FileParser object's path_type, true_class, and image_path instance variables lists into a pandas DataFrame and
            exports it to a .csv ffile

            Args:
                - name: {
                    type: str
                    description: The string denoting the name of the .csv file being written to. This assumes that '.csv' is present in
                                 name.
                }

                - val: {
                    type: bool
                    description: Boolean operator that togggles whether the parser generates a .csv file

                }
            
            Returns: None
        """
        if val:
            df = pd.DataFrame({'path_type':self.path_type, 'true_class':self.true_class, 'image_path':self.true_class})
            df.to_csv(name, index=False)
            print(f'"{name}" successfully written.')
        
    
    def mkdirs(self):
        """ Makes a directory to store the parsed image data. Also makes subsequent subdirectories for training, validation, and testing. 

        Args: None

        Returns: None
        """
        if not os.path.exists(self.push_path):
            os.makedirs(self.push_path)
            os.makedirs(self.push_path + '/train')
            os.makedirs(self.push_path + '/val')
            os.makedirs(self.push_path + '/test')
            print(f"Directory '{self.push_path}' and subdirectories created.")
        else:
            print(f"Directory '{self.push_path}' already exists.")

    
def main():
    # Pull path is where images are coming from; push path is where theya re going after being parsed
    pull_path = 'animals/'
    push_path = 'TransformedBatch/'
    
    # Instantiate object
    parser = FileParser(pull_path=pull_path, push_path=push_path)

    # Make push path directory and its subdirectories if needed. Parse images. 
    parser.mkdirs()
    parser.parse()

    # Write to .csv
    csv_name = 'all_animals.csv'
    parser.write_csv(csv_name)
    print("Parsing complete.")
    print('*'*60)


if __name__ == '__main__':
    main()
