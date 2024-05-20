#!/usr/bin/env python3
"""
Class RNNEncoder that inherits from tensorflow.keras.layers.Layer
to encode for machine translation
"""
import tensorflow as tf
import numpy as np


class RNNEncoder(tf.keras.layers.Layer):
    """
    Class RNNEncoder that inherits from tensorflow.keras.layers.Layer
    to encode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True)

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        """
        shape = (self.batch, self.units)
        tensor = tf.zeros(shape=shape)
        return tensor

    def call(self, x, initial):
        """
        Method to call the instance
        Arguments:
            - x is a tensor of shape (batch, input_seq_len) containing the
            input to the encoder layer as word indices within the vocabulary
            - initial is a tensor of shape (batch, units) containing the
            initial hidden state
        Returns: outputs, hidden
            - outputs is a tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder
            - hidden is a tensor of shape (batch, units) containing the last
            hidden state of the encoder
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(
            inputs=x,
            initial_state=initial)
        return outputs, hidden
