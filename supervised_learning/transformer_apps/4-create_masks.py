#!/usr/bin/env python3
"""Creates all masks for training/validation"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation

    Arguments:
        - inputs: tf.Tensor of shape (batch_size, seq_len_in)
        - target: tf.Tensor of shape (batch_size, seq_len_out)

    Returns:
        encoder_mask: (batch_size, 1, 1, seq_len_in)
        combined_mask: (batch_size, 1, seq_len_out, seq_len_out)
        decoder_mask: (batch_size, 1, 1, seq_len_in)
    """
    batch_size, seq_len_in = inputs.shape
    seq_len_out = target.shape[1]

    # Encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = tf.reshape(encoder_mask,
                              shape=(batch_size, 1, 1, seq_len_in))

    # Decoder target padding mask + lookahead mask
    decoder_target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoder_target_mask = tf.reshape(decoder_target_mask,
                                     shape=(batch_size, 1, 1, seq_len_out))

    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)

    combined_mask = tf.maximum(look_ahead_mask, decoder_target_mask)

    # Decoder mask (used in second attention block)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = tf.reshape(decoder_mask,
                              shape=(batch_size, 1, 1, seq_len_in))

    return encoder_mask, combined_mask, decoder_mask
