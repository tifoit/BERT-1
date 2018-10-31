import tensorflow as tf
from .layer_norm import LayerNormalization

class SublayerConnection(tf.layers.Layer):
    """
    Implements layer normalization + sublayer + dropout.
    """
    def __init__(self, sublayer, hidden_size, dropout, train=True):
        super(ResidualConnection, self).__init__()
        self.sublayer = sublayer
        self.norm = LayerNorm(hidden_size)
        self.dropout = dropout
        self.train = train

    def call(self, x):
        """
        Wraps a (feed forward / self-attention) layer.

        Args:
        x: Tensor, input to perform sublayer connection on

        Returns:
        Output of input after it goes through sublayer.
        """
        if self.train:
            return x + tf.nn.dropout(self.sublayer(self.norm(x)), 1 - self.dropout)
        return x + self.sublayer(self.norm(x))