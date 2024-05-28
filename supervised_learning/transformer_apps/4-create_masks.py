#!/usr/bin/env python3
""" Creates all masks for training/validation """
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation
    Arguments:
        - inputs is a tf.Tensor of shape (batch_size, seq_len_in) that
          contains the input sentence
        - target is a tf.Tensor of shape (batch_size, seq_len_out) that
          contains the target sentence
    Returns:
        encoder_mask, combined_mask, decoder_mask
            - encoder_mask is the tf.Tensor padding mask of shape
              (batch_size, 1, 1, seq_len_in) to be applied in the encoder
            - combined_mask is the tf.Tensor of shape
              (batch_size, 1, seq_len_out, seq_len_out) used in the 1st
              attention block in the decoder to pad and mask future tokens
              in the input received by the decoder with the look ahead mask
              (always ignore pad tokens)
            - decoder_mask is the tf.Tensor padding mask of shape
              (batch_size, 1, 1, seq_len_in) used in the 2nd attention block
              in the decoder
    """

    # Extract batch size and sequence lengths from inputs and target tensors
    batch_size, seq_len_in = inputs.shape
    seq_len_out = target.shape[1]

    # Encoder mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), dtype=tf.float32)
    # Reshape the encoder mask to (batch_size, 1, 1, seq_len_in) for
    # compatibility with attention mechanisms
    encoder_mask = tf.reshape(encoder_mask,
                              shape=(batch_size, 1, 1, seq_len_in))

    # Combined mask
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)
    decoder_target_mask = tf.cast(tf.math.equal(target, 0), dtype=tf.float32)
    # Reshape the decoder target mask to (batch_size, 1, 1, seq_len_out) for
    # compatibility with attention mechanisms
    decoder_target_mask = tf.reshape(decoder_target_mask,
                                     shape=(batch_size, 1, 1, seq_len_out))

    combined_mask = tf.maximum(look_ahead_mask, decoder_target_mask)

    # Decoder mask
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), dtype=tf.float32)
    # Reshape the decoder mask to (batch_size, 1, 1, seq_len_in) for
    # compatibility with attention mechanisms
    decoder_mask = tf.reshape(decoder_mask,
                              shape=(batch_size, 1, 1, seq_len_in))

    return encoder_mask, combined_mask, decoder_mask
