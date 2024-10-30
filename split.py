import time
import os
import math
from PIL import Image
import pandas as pd

# Global variables
training_percentage = 0.7 # percentage of images that go to training set
val_percentage = 0.2 # percentage of images that go to validation set
test_percentage = 0.1 # percentage of images that go to testing set

# We use default ImageNet weights and ImageNet mean, std, and shape for transforms

# The recommended mean parameters for transforms with ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
# The recommended standard deviation parameters for transforms with ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225]
# determines the height and width of a transformed image.
RESIZE_SHAPE = (224, 224)


class DataPrimer:

    def __init__(self, pull_path, push_path) -> None:
        self.pull_path = pull_path
        self.push_path = push_path

        self.path_type = [] # train or val
        self.true_class = [] # true class type
        self.image_paths = [] # image titles
        self.embeddings = [] # embeddings
        self.num_images = 0

        
    def remove_ds(self, some_list):
        """ If the ds file from a mac exists, remove it from the os directory list"""
        my_copy = some_list.copy()

        # Delete the macOS .DS_Store if it exists
        if ".DS_Store" in my_copy:
            my_id = my_copy.index(".DS_Store")
            del my_copy[my_id]

        return my_copy


    def image_prep(self):
        """ Preps the images for machine learning by cropping out image scales and adding RGB values"""
        
        # Get directory. With macs, there is a .ds directory. This also removes this directory if it is present
        parent_dir = self.remove_ds(os.listdir(self.pull_path))

        # We need to create two directories: train directory and validation directory
        paths = ["train", "val", "test"]

        # This creates any directories that may not already be present. Both the test folder and val folder will
        # contain all class folders but will not yet contain any images
        for path in paths:
            # Make train and val directories
            p = self.push_path + path
            if not os.path.exists(p):
                os.mkdir(p)

            # Make class directories within both train and val
            for child_dir in parent_dir:
                class_dir = p + "/" + child_dir
                if not os.path.exists(class_dir):
                    os.mkdir(class_dir)

        # Make a list of every image path
        for child in parent_dir:
            child_path = self.pull_path + child + '/'
            child_dir = os.listdir(child_path)
            for image in child_dir:
                self.num_images += 1

        print(f"Total number of Images: {self.num_images}")
        print('-'*60)
        successful_filereads = self.num_images
        species_left = len(parent_dir)

        
        # For each species, we will do the same type of processing. It makes keeping track easier
        global_time = time.time()
        for child in parent_dir:
            start_time = time.time()
            child_path = self.pull_path + child + '/'
            child_dir = os.listdir(child_path)

            child_dir_nfiles = len(child_dir)

            train_val_split_idx = int(math.floor(child_dir_nfiles * training_percentage))
            val_test_split_idx = train_val_split_idx + int(math.floor(child_dir_nfiles * val_percentage))
            
            # Append to true class
            for image in child_dir:
                self.true_class.append(child)

            # Split for training and validation
            train_images = child_dir[:train_val_split_idx]
            validation_images = child_dir[train_val_split_idx:val_test_split_idx]
            test_images = child_dir[val_test_split_idx:]
            
            # Training is 70 percent
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


            print(f"70 Percent Finished with {child}")
            time_elapsed = time.time() - start_time
            print("Elapsed time: " + time.strftime("%H:%M:%S.{}".format(str(time_elapsed % 1)[2:])[:15],
                                            time.gmtime(time_elapsed)))
            print()
            
            # Validation is 20 percent
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
            
            print(f"90 Percent Finished with {child}")
            time_elapsed = time.time() - start_time
            print("Elapsed time: " + time.strftime("%H:%M:%S.{}".format(str(time_elapsed % 1)[2:])[:15],
                                            time.gmtime(time_elapsed)))
            print()

            # Testing is 10 percent
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

            print(f"100 Percent Finished with {child}")
            time_elapsed = time.time() - start_time
            print("Elapsed time: " + time.strftime("%H:%M:%S.{}".format(str(time_elapsed % 1)[2:])[:15],
                                            time.gmtime(time_elapsed)))
            print(f"Folder {child} Completed.")

            species_left -= 1
            print(f"Folders Left to Process: {species_left}")
            print('-' * 60)

        print("Finished with All")
        print(f"Successful file reads: {successful_filereads}")
        total_time_elapsed = time.time() - global_time
        print("Total Elapsed Time: " + time.strftime("%H:%M:%S.{}".format(str(time_elapsed % 1)[2:])[:15],
                                            time.gmtime(total_time_elapsed)))
        
    
    def create_dfs(self):

        # Main df
        path_type_df = pd.DataFrame(self.path_type, columns=['path_type'])
        true_class_df = pd.DataFrame(self.true_class, columns=['true_class'])
        image_paths = pd.DataFrame(self.image_paths, columns=['image_path'])
        main_df = pd.concat([path_type_df, true_class_df, image_paths], axis=1)

        return main_df


def main():
    # This is the main
    pull_path = 'animals/'
    push_path = 'TransformedBatch/'
    
    if not os.path.exists(push_path):
        os.makedirs(push_path)
        os.makedirs(push_path + '/train')
        os.makedirs(push_path + '/val')
        os.makedirs(push_path + '/test')
    
    # Generate object
    im_prep = DataPrimer(pull_path, push_path)

    # Prep image
    im_prep.image_prep()

    # get df
    main_df = im_prep.create_dfs()

    title = 'animals.csv'
    main_df.to_csv(title, index='False')

    print("df constructed and saved")


if __name__ == '__main__':
    main()


