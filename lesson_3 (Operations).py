import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# Operations
x = tf.range(start=5, limit=100, delta=2, dtype=float)
x = tf.reshape(x, [4, -1])
y = tf.random.normal(x.get_shape(), mean=3, stddev=2)

z = tf.add(x, y)
print(z)
print("--------------------------------------")

z = tf.subtract(x, y)
print(z)
print("--------------------------------------")

z = tf.multiply(x, y)
print(z)
print("--------------------------------------")

z = tf.divide(x, y)
print(z)
print("--------------------------------------")
