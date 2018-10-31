import tensorflow as tf
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding

class BERTEmbedding(tf.layers.Layer):
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__(name="bert_embedding")
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = lambda x: tf.layers.dropout(x, rate=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)