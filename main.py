import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16))

loss = CategoricalCrossentropy(from_logits=True)
optimizer = Adam()
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print('model created and complied')
print(f'model summary:\n{model.summary()}')

data = np.loadtxt('input.csv', delimiter=',')

print('data loaded.')
X = data[:, :30_000] / 255.0
Y = data[:, 30_000:]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
print('train and test data splitted.')
x_train = x_train.reshape(len(x_train), 100, 100, 3)
x_test = x_test.reshape(len(x_test), 100, 100, 3)

batch_size = 100
epochs = 18
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

model.evaluate(x_test, y_test)
model.save('saved_model')
