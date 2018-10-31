import tensorflow as tf

class SegmentEmbedding(tf.layers.Layer):
    def __init__(self, embed_size=512):
        super().__init__(name="segment_embedding")
        self.num_embeddings = 3
        self.embedding_dim = embed_size
        self.padding_idx = 0

    def forward(self, x):
        raise NotImplementedError