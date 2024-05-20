#!/usr/bin/env python3
""" Scaled Dot Product Attention """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Scaled Dot Product Attention
    Arguments:
        - Q: tensor with shape (..., seq_len_q, dk) containing the query matrix
        - K: tensor with shape (..., seq_len_v, dk) containing the key matrix
        - V: tensor with shape (..., seq_len_v, dv) containing the value matrix
        - mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
            containing the optional mask, or defaulted to None
    Returns: output, weights
        - output: tensor with shape (..., seq_len_q, dv) containing the scaled
            dot product attention
        - weights: tensor with shape (..., seq_len_q, seq_len_v) containing
            the attention weights
    """
    # Perform the dot product between Q (queries) and K (keys) and transpose K
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Get the dimensionality of the keys
    keys_dimentionality = tf.cast(tf.shape(K)[-1], tf.float32)

    # Scale the dot product by the square root of dimensionality of the keys
    scaled_attention_logits = matmul_qk / tf.math.sqrt(keys_dimentionality)

    # Apply the mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax to get the attention weights
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply the weights by the values (V)
    output = tf.matmul(weights, V)

    # Return the attention output and the attention weights
    return output, weights
