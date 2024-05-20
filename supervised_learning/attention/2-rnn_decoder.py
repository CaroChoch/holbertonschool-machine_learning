#!/usr/bin/env python3
""" Class RNNDecoder to decode for machine translation """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ Class RNNDecoder to decode for machine translation """

    def __init__(self, vocab, embedding, units, batch):
        """ Class constructor
        Arguments:
            - vocab is integer representing the size of the output vocabulary
            - embedding is an integer representing the dimensionality of the
              embedding vector
            - units is an integer representing the number of hidden units in
              the RNN cell
            - batch is an integer representing the batch size
        """
        super(RNNDecoder, self).__init__()

        # Embedding layer to convert words from the vocabulary into embedding
        # vectors
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding)

        # GRU layer with 'units' hidden units initialized with 'glorot_uniform'
        self.gru = tf.keras.layers.GRU(
            units=units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True)

        # Dense layer to output the vocabulary size
        self.F = tf.keras.layers.Dense(vocab)

        # SelfAttention layer to calculate attention for machine translation
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """ Method to call the instance
        Arguments:
            - x is a tensor of shape (batch, 1) containing the previous word
              in the target sequence as an index of the target vocabulary
            - s_prev is a tensor of shape (batch, units) containing the
              previous decoder hidden state
            - hidden_states is a tensor of shape (batch, input_seq_len, units)
              containing the outputs of the encoder
        Returns: y, s
            - y is a tensor of shape (batch, vocab) containing the output word
              as a one hot vector in the target vocabulary
            - s is a tensor of shape (batch, units) containing the new decoder
              hidden state
        """
        # Calculate the context vector using self-attention mechanism
        context, _ = self.attention(s_prev, hidden_states)

        # Get the embedding for the input word
        x = self.embedding(x)

        # Concatenate the context vector with the input embedding
        context_expanded = tf.expand_dims(
            context,
            axis=1)  # (batch_size, 1, units)
        x = tf.concat(
            [context_expanded, x],
            axis=-1)  # (batch_size, 1, units + embedding_dim)

        # Pass the concatenated vector though the GRU layer
        output, s = self.gru(
            x, initial_state=s_prev)  # output shape: (batch_size, 1, units)

        # Reshape the output to match the dense layer input requirements
        batch_size = tf.shape(output)[0]  # get the batch size dynamically
        units = tf.shape(output)[2]  # get the units size dynamically
        output_reshaped = tf.reshape(
            output,
            (batch_size, units))  # (batch_size, units)

        # Apply the dense layer to get the final output word probabilities
        y = self.F(output_reshaped)  # shape: (batch_size, vocab)

        # Return the final output word and the new decoder hidden state
        return y, s
