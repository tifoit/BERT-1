base_params = {
    "embedder": Encoder,
    "embedder_params": {
        "sequence_length": 512
    }

    "transformer_block": {
        "initializer_gain": 1.0,
        # block layers
        "num_hidden_layers":12, 
        "hidden_size": 768,
        "layer_dropout": 0.1,
        # multiheaded self-attention
        "num_heads": 12,
        "attention_dropout": 0.1,
        # feed forward network
        "filter_size": 3072,
        "gelu_dropout": 0.1,
        "allow_ffn_pad": False
    }
}