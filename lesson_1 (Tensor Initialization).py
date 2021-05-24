import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

#Initialization of a tensor
x = tf.constant(4)
print(x)

x = tf.constant(4, shape=(1,1), dtype=tf.float32)
print(x)

x = tf.constant([[1,2,3], [4,5,6]])
print(x)

x = tf.reshape(x, [3,-1])
print(x)