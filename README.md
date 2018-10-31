# BERT: Bidirectional Embedding Representations from Transformers
This is my implementation of Google AI's BERT model ([paper](https://arxiv.org/pdf/1810.04805.pdf)), with the specific use case of Question-Answering in mind. The official repository is [here](https://github.com/google-research/bert) - I intend to develop a better grasp of Tensorflow and the different practices of training LM's introduced in the paper.

No code is borrowed from the official BERT repository, but only its architecture and new training methods.
<<<<<<< HEAD

## Architecture
- [ ] Embedding
    - [ ] Token embeddings: WordPiece
    - [ ] Segment embeddings
    - [x] Position embeddings
- [x] Encoder
    - [x] Stacked Transformer Encoders
        - [x] Self-attention
        - [x] Feed-forward network
        - [x] Layernormalization
        - [x] Residual sublayer connection

## Pre-Training
- [ ] Masked LM
- [ ] Sentence Prediction

## Task Fine-tuning
- [ ] SQuAD
=======
>>>>>>> 0838545552b2802e7bb86ac432a910a52e70794f
