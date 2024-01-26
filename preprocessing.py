"""
    importing stuff:
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import csv

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
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    # Iterate over all files and directories in the given directory
    for root, dirs, files in os.walk(search_directory):
        try:
            file_dir = int(root.replace(search_directory, ''))
            image_per_class[file_dir] = 0
            # Check if the file has an image extension
        except:
            continue

        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
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


def create_csv(search_directory, output_csv):
    image_extensions = ['.jpg']
    data_csv = []
    for root, dirs, files in os.walk(search_directory):
        classes = np.zeros(16)
        try:
            file_dir = int(root.replace(search_directory, ''))
            image_per_class[file_dir] = 0
            classes[file_dir] = 1
        except:
            continue

        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                im = np.array(Image.open(f'{root}\\{file}')).flatten()
                if len(im) != 30_000:
                    print(f'image {root}\\{file} has shape : {len(im)}', file=sys.stderr)
                    # Expand dimensions to make it 3-channel
                    im = np.reshape(im, (100, 100))
                    expanded_image_array = np.expand_dims(im, axis=2)

                    # Repeat the grayscale channel along the third axis
                    im = np.repeat(expanded_image_array, 3, axis=2)
                    plt.imshow(im)
                    plt.show()
                    im = im.flatten()
                    # Convert the NumPy array back to a Pillow image
                data_csv.append([*im, *classes])
                del im
    for line in data_csv:
        if len(line) != 30_016:
            raise Exception(f'line has length {len(line)}')
    with open(output_csv, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the data to the CSV file
        csv_writer.writerows(data_csv)

    print(f"CSV file '{output_csv}' created successfully. it has {len(data_csv)} rows and {len(data_csv[0])} columns.")


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0, 1],
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
)

read_images("images\\")
create_csv("images\\", 'input.csv')
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
    if j == 9:
        continue
    directory = f'images_original\\{j}'
    save_to_dir = f'images\\{j}'
    i = 0
    for batch in datagen.flow_from_directory(directory=directory, target_size=(100, 100), batch_size=1,
                                             save_to_dir=save_to_dir, save_format='jpg'):
        i += 1
        if i == 400:
            break
    resize_and_save(directory + f'\\{j}', save_to_dir)"""
