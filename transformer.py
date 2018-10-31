import tensorflow as tf
from attention.attention import MultiHeadedSelfAttention
from attention.utils import get_padding, get_padding_bias
from utils.feed_forward import FeedForwardNetwork
from utils.layer_norm import LayerNormalization
from utils.sublayer import SublayerConnection

class TransformerStack(tf.layers.Layer):
    """
    Stack of Transformer Encoder blocks.
    """
    def __init__(self, params, train=True):
        super(TransformerStack, self).__init__()
        self.layers = []
        self.params = params
        for _ in range(params["num_hidden_layers"]):
            # Create sublayers for each layer.
            self_attention_layer = MultiHeadedSelfAttention(
                params["num_heads"], params["hidden_size"],
                params["attention_dropout"], train)
            feed_forward_network = FeedForwardNetwork(
                params["hidden_size"], params["filter_size"],
                params["gelu_dropout"], train, params["allow_ffn_pad"])
            self.layers.append([
                SublayerConnection(self_attention_layer, params["hidden_size"], params["layer_dropout"], train),
                SublayerConnection(feed_forward_network, params["hidden_size"], params["layer_dropout"], train)])

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params["hidden_size"])

    def __call__(self, inputs):
        """
        Return the output of the transformer encoder layers.

        Args:
        inputs: int tensor with shape [batch_size, input_length]

        Returns:
        Output of encoder layer stack.
        float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        # Variance scaling is used here because it seems to work in many problems.
        initializer = tf.variance_scaling_initializer(
                self.params["initializer_gain"], mode="fan_avg", distribution="uniform")
        # Begin Transformer Encoder.
        with tf.variable_scope("transformer_encoder", initializer=initializer):
            # Calculate attention bias for encoder self-attention.
            attention_bias = get_padding_bias(inputs)
            # Calculate input padding for feed forward network.
            input_padding = get_padding(inputs)

            if self.train:
                inputs = tf.nn.dropout(inputs, 1 - self.params["layer_dropout"])

            for n, layer in enumerate(self.layers):
                # Run inputs through the sublayers.
                self_attention_layer = layer[0]
                feed_forward_network = layer[1]

                with tf.variable_scope("layer_%d" % n):
                    with tf.variable_scope("self_attention"):
                        inputs = self_attention_layer(inputs, attention_bias)
                    with tf.variable_scope("ffn"):
                        inputs = feed_forward_network(inputs, inputs_padding)

            outputs = self.output_normalization(inputs)
            return outputs