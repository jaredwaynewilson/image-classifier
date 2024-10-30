import os
import string
import random
from PIL import Image

# initializing size of string
N = 12

# Removes .DS file if it exists
def remove_ds(some_list):
        """ If the ds file from a mac exists, remove it from the os directory list"""
        my_copy = some_list.copy()

        # Delete the macOS .DS_Store if it exists
        if ".DS_Store" in my_copy:
            my_id = my_copy.index(".DS_Store")
            del my_copy[my_id]

        return my_copy


# Make animals dir
adam_path = 'animals'
if not os.path.exists(adam_path):
    os.mkdir(adam_path)

# Make subdirs for each animal type
animal_folders = ['cat', 'dog', 'squirrel']
for folder in animal_folders:
    seth_path = adam_path + '/' + folder
    if not os.path.exists(seth_path):
        os.mkdir(seth_path)

# Load images into subdirs.

# Iterate through names of animals
for animal_type in animal_folders:
    pull_path = animal_type

    breed_list = os.listdir(pull_path)
    breed_list = remove_ds(breed_list)

    # Iterate through breed subdirectories
    for breed in breed_list:
        breed_path = pull_path + '/' + breed
        image_set = os.listdir(breed_path)
        image_set = remove_ds(image_set)

        # Iterate through images in each breed subdirectory. Export image to new directory with name contraining random string and breed identifier
        for image in image_set:
            image_path = breed_path + '/' + image
            # Convert CYMK to RGB
            my_img = Image.open(image_path)
            my_img = my_img.convert('RGB')
            
            # Save
            random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
            export_name = random_string +'___' + breed + '.jpg'
            push_path = adam_path + '/' + animal_type + '/' + export_name
            my_img.save(push_path)