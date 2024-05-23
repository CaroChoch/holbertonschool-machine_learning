#!/usr/bin/env python3
""" Class EncoderBlock that creates an encoder block for a transformer """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ Class EncoderBlock """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        Arguments:
            - dm: Dimensionality of the model
            - h: Number of heads
            - hidden: Number of hidden units in the fully connected layer
            - drop_rate: Dropout rate
        Parameters:
            - mha: MultiHeadAttention layer
            - dense_hidden: Hidden dense layer with hidden units and relu
                activation
            - dense_output: Output dense layer with dm units
            - layernorm1: First layer normalization layer, with epsilon=1e-6
            - layernorm2: Second layer normalization layer, with epsilon=1e-6
            - dropout1: First dropout layer
            - dropout2: Second dropout layer
        """
        super(EncoderBlock, self).__init__()

        # Multi-head attention
        self.mha = MultiHeadAttention(dm, h)
        # Hidden layer
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # Output layer
        self.dense_output = tf.keras.layers.Dense(dm)
        # Layer Normalization 1
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Layer Normalization 2
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Dropout 1
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        # Dropout 2
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Method to call the instance
        Arguments:
            - x: Tensor of shape (batch, input_seq_len, dm) containing
                the input to the encoder block
            - training: Boolean to determine if the model is training
            - mask: Mask to be applied for multi-head attention
        Returns: A tensor of shape (batch, input_seq_len, dm) containing
            the block’s output
        """
        # Multi-head attention
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        # Add and Norm
        normed_attention_output = self.layernorm1(x + attention_output)
        # Feed forward
        feedforward_output = self.dense_hidden(normed_attention_output)
        feedforward_output = self.dense_output(feedforward_output)
        feedforward_output = self.dropout2(
            feedforward_output, training=training)
        # Add and Norm
        encoder_output = self.layernorm2(
            normed_attention_output + feedforward_output)

        return encoder_output
