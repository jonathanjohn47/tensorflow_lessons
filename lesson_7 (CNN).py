import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

model = keras.Sequential()
model.add(keras.Input(shape=x_train[0].shape))
model.add(layers.Conv2D(x_train[0].shape[0], (3, 3), activation='relu'))  # setting up kernel
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(x_train[0].shape[0] * 2, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr=3e-4), metrics=['accuracy'], )

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

print(model.summary())
