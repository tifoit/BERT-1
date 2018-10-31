import tensorflow as tf
import math

class GELU(tf.layers.Layer):
    """
    Section 3.4: use GELU instead of RELU
    """
    def call(self, x):
        return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))