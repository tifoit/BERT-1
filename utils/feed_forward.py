import tensorflow as tf
from .gelu import GELU


class FeedForwardNetwork(tf.layers.Layer):
  """
  Fully connected feedforward network.
  """

  def __init__(self, hidden_size, filter_size, gelu_dropout=0.1, train=True, allow_pad=False):
    super(FeedFowardNetwork, self).__init__(name="ffn")
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.gelu_dropout = gelu_dropout
    self.train = train
    self.allow_pad = allow_pad
    self.activation = GELU()

    self.filter_dense_layer = tf.layers.Dense(
        hidden_size, use_bias=True, activation=self.activation, name="filter_layer")
    self.output_dense_layer = tf.layers.Dense(
        filter_size, use_bias=True, name="output_layer")

  def call(self, x, padding=None):
    """
    Return outputs of the feedforward network.

    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      padding: (optional) If set, the padding values are temporarily removed
        from x (provided self.allow_pad is set). The padding values are placed
        back in the output tensor in the same locations.
        shape [batch_size, length]
        
    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    padding = None if not self.allow_pad else padding

    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    if padding is not None:
      with tf.name_scope("remove_padding"):
        # Flatten padding to [batch_size*length]
        pad_mask = tf.reshape(padding, [-1])
        nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

        # Reshape x to [batch_size*length, hidden_size] to remove padding
        x = tf.reshape(x, [-1, self.hidden_size])
        x = tf.gather_nd(x, indices=nonpad_ids)

        # Reshape x from 2 dimensions to 3 dimensions.
        x.set_shape([None, self.hidden_size])
        x = tf.expand_dims(x, axis=0)

    output = self.filter_dense_layer(x)
    if self.train:
      output = tf.nn.dropout(output, 1.0 - self.gelu_dropout)
    output = self.output_dense_layer(output)

    if padding is not None:
      with tf.name_scope("re_add_padding"):
        output = tf.squeeze(output, axis=0)
        output = tf.scatter_nd(
            indices=nonpad_ids,
            updates=output,
            shape=[batch_size * length, self.hidden_size]
        )
        output = tf.reshape(output, [batch_size, length, self.hidden_size])
    return output
