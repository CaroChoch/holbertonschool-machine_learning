#!/usr/bin/env python3
"""TF encode method"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """Dataset class"""

    def __init__(self):
        """Class constructor"""
        self.data_train, self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for dataset using numpy generator
        to avoid timeout
        """
        pt_sentences = []
        en_sentences = []

        for pt, en in tfds.as_numpy(data):
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_sentences, target_vocab_size=2 ** 15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_sentences, target_vocab_size=2 ** 15
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        """
        pt_start = self.tokenizer_pt.vocab_size
        pt_end = pt_start + 1
        en_start = self.tokenizer_en.vocab_size
        en_end = en_start + 1

        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = [pt_start] + self.tokenizer_pt.encode(pt_text) + [pt_end]
        en_tokens = [en_start] + self.tokenizer_en.encode(en_text) + [en_end]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper around self.encode
        """
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
