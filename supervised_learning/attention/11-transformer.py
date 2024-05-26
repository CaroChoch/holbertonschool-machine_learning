#!/usr/bin/env python3
""" Transformer Network """
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """ Transformer class """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor
        Arguments:
            - N: Number of blocks in the encoder and decoder
            - dm: Dimensionality of the model
            - h: Number of heads
            - hidden: Number of hidden units in the fully connected layers
            - input_vocab: Size of the input vocabulary
            - target_vocab: Size of the target vocabulary
            - max_seq_input: Maximum sequence length possible for the input
            - max_seq_target: Maximum sequence length possible for the target
            - drop_rate: Dropout rate
        """
        super(Transformer, self).__init__()
        # Encoder layer
        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate)
        # Decoder layer
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate)
        # Final linear layer
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Method to call the transformer
        Arguments:
            - inputs: Tensor of shape (batch, input_seq_len) containing
                the inputs
            - target: Tensor of shape (batch, target_seq_len) containing
                the target
            - training: Boolean to determine if the model is training
            - encoder_mask: Padding mask to be applied to the encoder
            - look_ahead_mask: Mask to be applied to the decoder
            - decoder_mask: Padding mask to be applied to the decoder
        Returns:
            - Tensor of shape (batch, target_seq_len, target_vocab) containing
                the transformer output
        """
        # Encoder output
        enc_output = self.encoder(inputs, training, encoder_mask)
        # Decoder output
        dec_output = self.decoder(
            target,
            enc_output,
            training,
            look_ahead_mask,
            decoder_mask)
        # Linear layer
        final_output = self.linear(dec_output)
        # Return final output
        return final_output
