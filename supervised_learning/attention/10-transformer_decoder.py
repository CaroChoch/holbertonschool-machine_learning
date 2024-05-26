#!/usr/bin/env python3
""" Transformer Decoder """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """ Decoder class """
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class constructor
        Arguments:
            - N: Number of blocks in the decoder
            - dm: Dimensionality of the model
            - h: Number of heads in the multi-head attention mechanism
            - hidden: Number of hidden units in the fully connected layer
            - target_vocab: Size of the target vocabulary
            - max_seq_len: Maximum sequence length possible
            - drop_rate: Dropout rate
        """
        super(Decoder, self).__init__()
        self.dm = dm
        self.N = N
        # Embedding layer for target vocabulary
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        # Positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # Create and append each decoder block
        self.blocks = []
        for _ in range(N):
            self.blocks.append(DecoderBlock(dm, h, hidden, drop_rate))
        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Method to call the decoder
        Arguments:
            - x: Tensor of shape (batch_size, target_seq_len, dm) containing
                the input to the decoder
            - encoder_output: Tensor of shape (batch_size, input_seq_len, dm)
                containing the output of the encoder
            - training: Boolean to determine if the model is training
            - look_ahead_mask: Mask to be applied to the first multi-head
                attention layer
            - padding_mask: Mask to be applied to the second multi-head
                attention layer
        Returns:
            - Tensor of shape (batch_size, target_seq_len, dm) containing the
                decoder output
        """
        seq_len = x.shape[1]
        x = self.embedding(x)  # Apply embedding to input
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))  # Scale embeddings
        x += self.positional_encoding[:seq_len]  # Add positional encoding
        x = self.dropout(x, training=training)  # Apply dropout

        for i in range(self.N):  # Apply each decoder block
            x = self.blocks[i](
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask)

        # Return the decoder output
        return x
