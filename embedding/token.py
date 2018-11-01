import tensorflow as tf

class TokenEmbedding(object):
    def __init__(self, vocab_size, embed_size=768):
        super().__init__(name="token_embedding")
        self.num_embeddings = vocab_size
        self.embedding_dim = embed_size
        self.padding_idx = 0

    def forward(self, x):
        # raise NotImplementedError
        return None