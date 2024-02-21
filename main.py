import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, AveragePooling2D
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from keras.regularizers import l2, l1
from sklearn.model_selection import train_test_split

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(200, 200, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add((AveragePooling2D(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

loss = BinaryCrossentropy(from_logits=False)
optimizer = Adam(learning_rate=0.0001)
metrics = ['accuracy', Precision(), Recall()]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print('model created and complied')
print(f'model summary:\n{model.summary()}')
for i in range(15, 16):

    train_data = np.loadtxt(f'D:\\new data set\\{i}_train.csv', delimiter=',')
    print(f'class {i} train data loaded -> train: {train_data.shape}')
    x_train = train_data[:, :120_000] / 255.0
    y_train = train_data[:, 120_000:]
    del train_data
    x_train = x_train.reshape(len(x_train), 200, 200, 3)
    print(f'train data : {x_train.shape}\ny train : {y_train.shape}\n')
    batch_size = 100
    epochs = 20
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    print('training completed, loading the test data:')
    test_data = np.loadtxt(f'D:\\new data set\\{i}_test.csv', delimiter=',')
    print(f'test data loaded -> test: {test_data.shape}')
    x_test = test_data[:, :120_000] / 255.0
    y_test = test_data[:, 120_000:]
    del test_data
    x_test = x_test.reshape(len(x_test), 200, 200, 3)
    print(f'test data: {x_test.shape}\ny test: {y_test.shape}\n')
    model.evaluate(x_test, y_test)
    print(f'that was class {i}, saving the model in D:\\new data set\\saved_models\\saved_model_{i}.')
    model.save(f'D:\\new data set\\saved_models\\saved_model_{i}')
    del x_train, x_test, y_train, y_test
    model.reset_states()
