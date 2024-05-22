#!/usr/bin/enV python3
""" Multi Head Attention """
import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Multi Head Attention """

    def __init__(self, dm, h):
        """
        Class constructor
        Arguments:
            - dm is an integer representing the dimensionality of the model
            - h is an integer representing the number of heads
            - dm is diVisible by h
                * h - the number of heads
                * dm - the dimensionality of the model
                * depth - the depth of each attention head
                * WQ - a Dense layer with dm units, used to generate the
                    Query matrix
                * WK - a Dense layer with dm units, used to generate the
                    Key matrix
                * WV - a Dense layer with dm units, used to generate the
                    Value matrix
                * linear - a Dense layer with dm units, used to generate
                    the attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        # Dense layer used to generate the Query matrix
        self.Wq = tf.keras.layers.Dense(dm)
        # Dense layer used to generate the Key matrix
        self.Wk = tf.keras.layers.Dense(dm)
        # Dense layer used to generate the Value matrix
        self.Wv = tf.keras.layers.Dense(dm)

        # Dense layer used to generate the attention output
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension
        Arguments:
            - x is a tensor of shape (batch, seQ_len, dm)
                containing the input to split
            - batch_size is an integer representing the batch size
        Returns: a tensor with shape (batch, h, seQ_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Call Method
        Arguments:
            - Q is a tensor of shape (batch, seQ_len_Q, dm)
                containing the input to generate the Query matrix
            - K is a tensor of shape (batch, seQ_len_V, dm)
                containing the input to generate the Key matrix
            - V is a tensor of shape (batch, seQ_len_V, dm)
                containing the input to generate the Value matrix
            - masK is always None
        Returns: output, weights
            - output a tensor with its last two dimensions as (...,
                seQ_len_Q, dm)
                containing the scaled dot product attention
            - weights a tensor with its last three dimensions as
                (..., h, seQ_len_Q, seQ_len_V) containing the attention weights
        """
        batch_size = tf.shape(Q)[0]

        # Generate the Query, Key, and Value matrices
        Q = self.Wq(Q)  # (batch, seq_len_q, dm)
        K = self.Wk(K)  # (batch, seq_len_v, dm)
        V = self.Wv(V)  # (batch, seq_len_v, dm)

        # Split and transpose for multi-head attention
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled Dot Product Attention
        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        # Transpose and reshape the scaled_attention back
        # to (batch, seq_len_q, dm)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.dm))

        # Final linear layer
        output = self.linear(concat_attention)  # (batch, seq_len_q, dm)

        return output, weights
