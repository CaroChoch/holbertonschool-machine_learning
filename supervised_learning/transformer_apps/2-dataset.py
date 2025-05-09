#!/usr/bin/env python3
"""TF encode wrapper for TensorFlow."""

import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    """Dataset class for Portuguese-English translations."""

    def __init__(self):
        """Initialize the Dataset, build tokenizers, and encode data."""
        # Load the TED talk translation dataset
        self.data_train, self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True
        )

        # Build subword tokenizers using a subset of the training data to speed up
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        # Apply the TensorFlow wrapper for encoding
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Create SubwordTextEncoder tokenizers for Portuguese and English
        using only the first 10,000 examples to avoid timeout.
        """
        # Generators for a subset of sentences
        pt_gen = (
            pt.numpy().decode('utf-8')
            for pt, _ in tfds.as_numpy(data.take(10_000))
        )
        en_gen = (
            en.numpy().decode('utf-8')
            for _, en in tfds.as_numpy(data.take(10_000))
        )

        # Build the tokenizers from the corpus generators
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_gen, target_vocab_size=2**15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_gen, target_vocab_size=2**15
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encode a pair of sentences into token IDs, including start and end tokens.
        """
        # Define start and end token IDs for each language
        pt_start = self.tokenizer_pt.vocab_size
        pt_end = pt_start + 1
        en_start = self.tokenizer_en.vocab_size
        en_end = en_start + 1

        # Decode bytes to strings
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Encode to token IDs and add start/end tokens
        pt_tokens = [pt_start] + self.tokenizer_pt.encode(pt_text) + [pt_end]
        en_tokens = [en_start] + self.tokenizer_en.encode(en_text) + [en_end]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper around the encode method, returning TensorFlow tensors.
        """
        # Use py_function to call the Python encode method
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        # Set static shape for the tensors (variable length)
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
