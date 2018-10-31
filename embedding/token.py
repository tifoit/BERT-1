import tensorflow as tf

class TokenEmbedding(tf.layers.Layer):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(name="token_embedding")
        self.num_embeddings = vocab_size
        self.embedding_dim = embed_size
        self.padding_idx = 0

    def forward(self, x):
        raise NotImplementedError