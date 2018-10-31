import tensorflow as tf
import math

class PositionalEmbedding(tf.layers.Layer):

    def __init__(self, d_model, max_len=512, min_timescale=1.0, max_timescale=1.0e4):
        super().__init__(name="positional_embedding")

        pe = tf.zeros((max_len, d_model), dtype=tf.float32)
        position = tf.expand_dims(tf.range(0, max_len, dtype=tf.float32), axis=0)
        div_term = tf.exp((tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(max_timescale) / d_model)))
        pe[:, 0::2] = tf.sin(position * div_term)
        pe[:, 1::2] = tf.cos(position * div_term)
        self.pe = tf.expand_dims(pe, axis=0)

    def forward(self, x):
        return self.pe[:, :tf.shape(x)[1]]