import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# Indexing of a Tensor
x = tf.range(start=4, limit=90, )
x = tf.reshape(x, [-1, 2])

y = x[0:6, :]
print(y)

x_ind = tf.gather(x, [4, 2])
print(x_ind)