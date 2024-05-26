#!/usr/bin/env python3
""" Transformer Encoder """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder
    """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Initialize the Transformer Encoder
        Arguments:
            - N: Number of blocks in the encoder
            - dm: Dimensionality of the model
            - h: Number of heads in the multi-head attention mechanism
            - hidden: Number of hidden units in the fully connected layer
            - input_vocab: Integer representing the size of the input
                vocabulary
            - max_seq_len: Integer representing the maximum sequence length
            - drop_rate: Dropout rate
        """
        super(Encoder, self).__init__()
        self.dm = dm
        self.N = N
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for _ in range(N):
            self.blocks.append(EncoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Call method for the Transformer Encoder
        Arguments:
            - x: Tensor of shape (batch, input_seq_len) containing the input
                to the encoder
            - training: Boolean to determine if the model is in training mode
            - mask: Mask to be applied for multi-head attention
        Returns:
            - Tensor of shape (batch, input_seq_len, dm) containing the encoder
                output
        """
        seq_len = x.shape[1]
        x = self.embedding(x)  # Apply embedding
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))  # Scale embeddings
        x += self.positional_encoding[:seq_len]  # Add positional encoding
        x = self.dropout(x, training=training)  # Apply dropout

        for i in range(self.N):  # Apply each encoder block
            x = self.blocks[i](x, training, mask)

        # Return the encoder output
        return x
