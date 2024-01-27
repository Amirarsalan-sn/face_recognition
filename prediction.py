from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

image_path = input('Enter the path to the desired image: ')
im = Image.open(image_path)
im = im.resize((100, 100))
im_copy = np.array(im)
if im_copy.shape != (100, 100, 3):
    expanded_image_array = np.expand_dims(im_copy, axis=2)
    im_copy = np.repeat(expanded_image_array, 3, axis=2)

im_copy = im_copy / 255.0
model = load_model('saved_model')
y_pred = model.predict(im_copy.reshape(1, 100, 100, 3))

#y_pred = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
result = []
for pred_index in range(len(y_pred[0])):
    result.append(f'{class_names[pred_index]} -> {y_pred[0][pred_index]}')

result = "\n".join(result)

print(result)
plt.imshow(im)
plt.title('input image')
plt.show()
