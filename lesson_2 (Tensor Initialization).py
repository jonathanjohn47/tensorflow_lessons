import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

#Some special initializtions

x = tf.zeros([3, 3])

x = tf.ones([2, 3])

x = tf.eye(3)  # I for identity matrix (eye)

x = tf.random.normal([3, 3], mean=0, stddev=1)
print(x)

x = tf.random.uniform([3,3], minval = 0, maxval=1)
print(x)

x = tf.range(9)
print(x)

x = tf.range(start = 4, limit = 10, delta = 2)
print(x)
