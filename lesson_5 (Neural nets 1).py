import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])

x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)

# Sequential API (Very convenient but not flexible)
print(x_train.shape)
print(y_train.shape)
# One way to add layers
model = keras.Sequential([
    keras.Input(shape=(x_train.shape[1] * x_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10),
])
print(model.summary())

# Second way to add layers
model = keras.Sequential()
model.add(keras.Input(shape=(x_train.shape[1],)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))

# Compile the model at the end
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
