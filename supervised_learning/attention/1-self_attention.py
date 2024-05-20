#!/usr/bin/env python3
""" Class SelfAttention to calculate attention for machine translation """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ Class SelfAttention to calculate attention for machine translation """

    def __init__(self, units):
        """ Class constructor """
        super(SelfAttention, self).__init__()
        # Weight matrix for yhe previous decoder hidden state
        self.W = tf.keras.layers.Dense(units)
        # Weight matrix for the encoder hidden states
        self.U = tf.keras.layers.Dense(units)
        # Weight matrix for the attention scoring
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Method to call the instance
        Arguments:
            - s_prev is a tensor of shape (batch, units) containing the
            previous decoder hidden state
            - hidden_states is a tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder
        Returns: context, weights
            - context is a tensor of shape (batch, units) that contains the
            context vector for the decoder
            - weights is a tensor of shape (batch, input_seq_len, 1) that
            contains the attention weights
        """
        # Adding a dimension to match hidden_states dimensions
        s_prev = tf.expand_dims(s_prev, 1)
        # Calculate the  attention scoring
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        # Calculate the attention weights
        weights = tf.nn.softmax(score, axis=1)
        # Apply the weights to the hidden states
        context = weights * hidden_states
        # Sum the weighted hidden states to get the context vector
        context = tf.reduce_sum(context, axis=1)

        # Return the context vector and the attention weights
        return context, weights
