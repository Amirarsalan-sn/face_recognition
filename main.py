"""
    importing stuff:
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

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


def read_images(directory):
    count = 0
    min_size = (1e10, 1e10)
    min_size_addr = None
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    # Iterate over all files and directories in the given directory
    for root, dirs, files in os.walk(directory):
        try:
            file_dir = int(root.replace(directory, ''))
            image_per_class[file_dir] = 0
            # Check if the file has an image extension
        except:
            continue

        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
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


"""read_images("images\\")
im = Image.open("images\\8\\1.jpg")
plt.figure()
plt.subplot(1, 2, 1)
plt.title(f"shape : {im.size}")
plt.imshow(im)

im2 = im.resize((100, 100))
plt.subplot(1, 2, 2)
plt.title(f"shape : {im2.size}")
plt.imshow(im2)

im3 = im.resize((200, 200))
plt.subplot(1, 2, 2)
plt.title(f"shape : {im3.size}")
plt.imshow(im3)

plt.show()
print(np.array(im))"""
#read_images("images\\")

im = Image.open("images\\10\\image.jpeg")
plt.figure(figsize=(2, 2))
plt.imshow(im)
plt.title(f'shape : {im.size}')
plt.show()