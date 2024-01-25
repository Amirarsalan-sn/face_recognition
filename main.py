"""
    importing stuff:
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

class_names = {
    0: 'Ali Day',
    1: "Mohsen Chavoshi",
    2: 'Mohamad Esfehani',
    3: 'Taraneh Alidostnia',
    4: 'Bahram Radan',
    5: 'Sogol Khaligh',
    6: 'Homayoon Shajarian',
    7: 'Sahar Dolatshahi',
    8: 'Mehran Ghafourian',
    9: 'Mehran Modiri',
    10: 'Reza Attaran',
    11: 'Javad Razavian',
    12: 'Seyed Jalal Hoseini',
    13: 'Alireza Beyranvand',
    14: 'Nazanin Bayati',
    15: 'Bahareh Kianafshar',
}

image_per_class = {}


def read_images(search_directory):
    count = 0
    min_size = (1e10, 1e10)
    min_size_addr = None
    skip_next = False
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    # Iterate over all files and directories in the given directory
    for root, dirs, files in os.walk(search_directory):
        if skip_next is True:
            skip_next = False
            continue
        try:
            file_dir = int(root.replace(search_directory, ''))
            image_per_class[file_dir] = 0
            # Check if the file has an image extension
        except:
            continue

        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                skip_next = True
                """im = Image.open(f'{root}\\{file}')
                if min(im.size) < min(min_size):
                    min_size = im.size
                    min_size_addr = f'{root}\\{file}'
                del im"""
                count += 1
                image_per_class[file_dir] += 1

    print('total count: ', count)
    print('image per class count: ')
    for cls in image_per_class.keys():
        print(f'{class_names[cls]} : {image_per_class[cls]}')

    print('min size: ', min_size)
    print('min size addr', min_size_addr)


def resize_and_save(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get a list of all files in the input directory
    files = os.listdir(input_directory)

    # Loop through each file in the input directory
    for file in files:
        # Open the image using PIL
        image = Image.open(os.path.join(input_directory, file))

        # Resize the image to 100x100 pixels
        resized_image = image.resize((100, 100))

        # Generate a unique filename for the output image
        output_filename = os.path.join(output_directory, f"{os.path.splitext(file)[0]}_resized.jpg")

        # Save the resized image with a unique name in the output directory
        resized_image.save(output_filename)

        print(f"Saved {output_filename}")

    print("Image resizing and saving complete.")


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0, 1],
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=2,
    horizontal_flip=True,
)


read_images("images\\")

"""im = np.array(Image.open('images\\8\\1.jpg'))
im = im.reshape((1,) + im.shape)
print(im.shape)
i = 0
for batch in datagen.flow(im, batch_size=20,
                          save_to_dir='images\\8\\test2', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break"""
"""for j in range(16):
    directory = f'images\\{j}'
    save_to_dir = f'images\\{j}'
    i = 0
    for batch in datagen.flow_from_directory(directory=directory, target_size=(100, 100), batch_size=1,
                                             save_to_dir=save_to_dir, save_format='jpg'):
        i += 1
        if i == 200:
            break
    resize_and_save(directory + f'\\{j}', save_to_dir)"""
