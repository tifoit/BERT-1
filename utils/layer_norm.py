import tensorflow as tf

class LayerNormalization(tf.layers.Layer):
  """
  Applies layer normalization.
  """

  def __init__(self, hidden_size, eps=1e-6):
    super(LayerNormalization, self).__init__(name="layer_norm")
    self.hidden_size = hidden_size
    self.eps = eps
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                initializer=tf.ones_initializer())
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer())
    self.built = True

  def call(self, x):
    """
    Computes and applies layer normalization.

    Args:
    x: Tensor to normalize

    Returns:
    Input with layer normalization applied.
    """
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + self.eps)
    return norm_x * self.scale + self.bias