#!/usr/bin/env python3
""" Class DecoderBlock that creates a decoder block for a transformer """

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ Class DecoderBlock """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        Arguments:
            - dm: Dimensionality of the model
            - h: Number of heads
            - hidden: Number of hidden units in the fully connected layer
            - drop_rate: Dropout rate
        Parameters:
            - mha1: First MultiHeadAttention layer (Masked Multi-Head
                Attention)
            - mha2: Second MultiHeadAttention layer (Encoder-Decoder Attention)
            - dense_hidden: Hidden dense layer with hidden units and
                relu activation
            - dense_output: Output dense layer with dm units
            - layernorm1: First layer normalization layer, with epsilon=1e-6
            - layernorm2: Second layer normalization layer, with epsilon=1e-6
            - layernorm3: Third layer normalization layer, with epsilon=1e-6
            - dropout1: First dropout layer
            - dropout2: Second dropout layer
            - dropout3: Third dropout layer
        """
        super(DecoderBlock, self).__init__()

        # First MultiHeadAttention layer (Masked Multi-Head Attention)
        self.mha1 = MultiHeadAttention(dm, h)
        # Second MultiHeadAttention layer (Encoder-Decoder Attention)
        self.mha2 = MultiHeadAttention(dm, h)
        # Hidden layer
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # Output layer
        self.dense_output = tf.keras.layers.Dense(dm)
        # Layer Normalization 1
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Layer Normalization 2
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Layer Normalization 3
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Dropout 1
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        # Dropout 2
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        # Dropout 3
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Method to call the instance
        Arguments:
            - x: Tensor of shape (batch, target_seq_len, dm) containing the
                input to the decoder block
            - encoder_output: Tensor of shape (batch, input_seq_len, dm)
                containing the output of the encoder
            - training: Boolean to determine if the model is training
            - look_ahead_mask: Mask to be applied to the first multi-head
                attention layer
            - padding_mask: Mask to be applied to the second multi-head
                attention layer
        Returns: A tensor of shape (batch, target_seq_len, dm) containing the
            block’s output
        """
        # First multi-head attention block (Masked Multi-Head Attention)
        masked_attention_output, _ = self.mha1(x, x, x, look_ahead_mask)
        masked_attention_output = self.dropout1(
            masked_attention_output,
            training=training)
        normed_masked_attention_output = self.layernorm1(
            x + masked_attention_output)

        # Second multi-head attention block (Encoder-Decoder Attention)
        enc_dec_attention_output, _ = self.mha2(
            normed_masked_attention_output,
            encoder_output,
            encoder_output,
            padding_mask)
        enc_dec_attention_output = self.dropout2(
            enc_dec_attention_output, training=training)
        normed_enc_dec_attention_output = self.layernorm2(
            normed_masked_attention_output + enc_dec_attention_output)

        # Feed forward neural network
        feed_forward_neural_output = self.dense_hidden(
            normed_enc_dec_attention_output)
        feed_forward_neural_output = self.dense_output(
            feed_forward_neural_output)
        feed_forward_neural_output = self.dropout3(
            feed_forward_neural_output,
            training=training)
        decoder_block_output = self.layernorm3(
            normed_enc_dec_attention_output + feed_forward_neural_output)

        return decoder_block_output
