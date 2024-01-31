"""
    importing stuff:
"""
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import csv
import imgaug as ia
import imgaug.augmenters as iaa

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
    valid_image_extensions = ['.jpg']
    invalid_image_extensions = ['.jpeg', '.png', '.gif', '.bmp']

    # Iterate over all files and directories in the given directory
    for root, dirs, files in os.walk(search_directory):
        try:
            file_dir = int(root.replace(search_directory, ''))
            image_per_class[file_dir] = 0
            # Check if the file has an image extension
        except:
            continue

        for file in files:
            if any(file.lower().endswith(ext) for ext in invalid_image_extensions):
                raise Exception(f'image {root}\\{file} has an invalid extension.')
            if any(file.lower().endswith(ext) for ext in valid_image_extensions):
                im = Image.open(f'{root}\\{file}')
                if min(im.size) < min(min_size):
                    min_size = im.size
                    min_size_addr = f'{root}\\{file}'
                del im
                count += 1
                image_per_class[file_dir] += 1

    print('total count: ', count)
    print('image per class count: ')
    for cls in image_per_class.keys():
        print(f'{class_names[cls]} : {image_per_class[cls]}')

    print('min size: ', min_size)
    print('min size addr', min_size_addr)


def resize_and_save(input_directory, output_directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    # Iterate over all files and directories in the given directory
    for root, dirs, files in os.walk(input_directory):

        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                im = Image.open(f'{root}\\{file}')
                if im.mode in ["L", "P"]:
                    im = im.convert("RGB")
                im = im.resize((200, 200))
                file_name = f'{output_directory}\\_{file}_resized.jpg'
                im.save(file_name)
                print(f"Saved {file_name}")

    print("Image resizing and saving complete.")


def create_csv(search_directory, output_csv):
    image_extensions = ['.jpg']
    rows = 0
    columns = 0
    for root, dirs, files in os.walk(search_directory):
        classes = np.zeros(16)
        try:
            file_dir = int(root.replace(search_directory, ''))
            classes[file_dir] = 1
        except:
            continue
        if file_dir == 9:
            continue
        data_csv = []
        print(f'class {file_dir}')
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                im = np.array(Image.open(f'{root}\\{file}')).flatten()
                if im.size != 120_000:
                    raise Exception(f'image {root}\\{file} doesn\'t have the right shape')
                # Convert the NumPy array back to a Pillow image
                data_csv.append([*im, *classes])
                del im
        with open(output_csv, 'a', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)

            # Write the data to the CSV file
            csv_writer.writerows(data_csv)

        rows += len(data_csv)
        columns = len(data_csv[0])
        del data_csv

    print(f"CSV file '{output_csv}' created successfully. it has {rows} rows and {columns} columns.")


def geometry_augment():
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0
    )

    for j in range(16):
        if j == 9:
            continue
        directory = f'color_augmented_images\\{j}'
        save_to_dir = f'train_images\\{j}'
        i = 0
        for batch in datagen.flow_from_directory(directory=directory, target_size=(200, 200), batch_size=1,
                                                 save_to_dir=save_to_dir, save_format='jpg', color_mode='rgb'):
            i += 1
            if i == 300:
                break
        resize_and_save(directory + f'\\{j}', save_to_dir)


def augment_each_class(num, directory, save_to_dir):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0
    )

    i = 0
    for batch in datagen.flow_from_directory(directory=directory, target_size=(200, 200), batch_size=1,
                                             save_to_dir=save_to_dir, save_format='jpg', color_mode='rgb'):
        i += 1
        if i == num:
            break
    resize_and_save(directory, save_to_dir)


def color_augment():
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    source_path = f'color_augmented_images\\'
    ia.seed(random.randint(1, 100))
    seq = iaa.Sequential([
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.7,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.6, 1.4), per_channel=1),
    ], random_order=True)

    for root, dirs, files in os.walk(source_path):
        images = []
        image_dir = False
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_dir = True
                im_np = Image.open(f'{root}\\{file}').resize((200, 200))
                if im_np.mode in ["L", "P"]:
                    im_np = im_np.convert("RGB")
                images.append(im_np)

        if not image_dir:
            continue
        images = np.array(images)
        images_aug = seq(images=images)
        i = 0
        for image in images_aug:
            im = Image.fromarray(image)
            im.save(f'{root}\\{i}_augmented.jpg')
            i += 1

        print(f'saved augmented images for {root}')


def create_test_csv(output_csv):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    source_path = f'test_images\\'
    csv_data = []
    for root, dirs, files in os.walk(source_path):
        classes = np.zeros(16)
        try:
            file_dir = int(root.replace(source_path, ''))
            classes[file_dir] = 1
        except:
            continue

        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                im = Image.open(f'{root}\\{file}')
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im = im.resize((200, 200))
                im_np = np.array(im).flatten()
                csv_data.append([*im_np, *classes])

    with open(output_csv, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the data to the CSV file
        csv_writer.writerows(csv_data)

    print(f'test file created with {len(csv_data)} rows and {len(csv_data[0])} columns.')


def create_csv_for_class(root_path, class_num, class_count):
    image_extensions = ['.jpg']
    source_path = f'{root_path}\\{class_num}\\'
    data_csv = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                im = Image.open(f'{root}\\{file}')
                im = np.array(im).flatten()
                if im.size != 120_000:
                    raise Exception(f'image {root}\\{file} doesn\'t have the right shape')
                data_csv.append([*im, 1])

    each_class = class_count // 14
    for i in range(16):
        if i == 9 or i == class_num:
            continue
        source_path = f'{root_path}\\{i}\\'
        indexes = random.sample(range(0, class_count + 70),
                                each_class)  # max count of a class without geometry augmentation is 70.
        indexes = sorted(indexes)
        j = 0
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if j not in indexes:
                    j += 1
                    continue
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    j += 1
                    im = Image.open(f'{root}\\{file}')
                    im = np.array(im).flatten()
                    if im.size != 120_000:
                        raise Exception(f'image {root}\\{file} doesn\'t have the right shape')
                    data_csv.append([*im, 0])

    random.shuffle(data_csv)
    with open(f'{root_path}\\{class_num}_train.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the data to the CSV file
        csv_writer.writerows(data_csv)

    print(f'{root_path}\\{class_num}_train.csv file created with {len(data_csv)} rows and {len(data_csv[0])} columns.')


def create_test_csv_for_class(dest_path, class_num):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    source_path = f'test_images\\{class_num}'
    data_csv = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                im = Image.open(f'{root}\\{file}')
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im = im.resize((200, 200))
                im_np = np.array(im).flatten()
                data_csv.append([*im_np, 1])

    remaining_test = len(data_csv)
    while remaining_test != 0:
        for root, dirs, files in os.walk('test_images\\'):
            try:
                file_dir = int(root.replace('test_images\\', ''))
            except:
                continue
            if file_dir == 9 or file_dir == class_num:
                continue
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    if random.randint(1, 5) == 1:
                        if remaining_test == 0:
                            break
                        remaining_test -= 1
                        im = Image.open(f'{root}\\{file}')
                        if im.mode != "RGB":
                            im = im.convert("RGB")
                        im = im.resize((200, 200))
                        im = np.array(im).flatten()
                        data_csv.append([*im, 0])

    random.shuffle(data_csv)
    with open(f'{dest_path}\\{class_num}_test.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the data to the CSV file
        csv_writer.writerows(data_csv)

    print(f'{dest_path}\\{class_num}_test.csv file created with {len(data_csv)} rows and {len(data_csv[0])} columns.')


# color_augment()
# geometry_augment()
# read_images("train_images\\")
# create_csv("train_images\\", 'train.csv')
# create_test_csv('test.csv')
"""for k in range(16):
    if k == 9:
        continue
    augment_each_class(1000, f'color_augmented_images\\{k}', f'D:\\images for the project\\1000_images\\{k}')"""
for k in range(16):
    if k == 9:
        continue
    create_csv_for_class('D:\\images for the project\\1000_images\\', k, 1000)
    create_test_csv_for_class('D:\\images for the project\\1000_images\\', k)
