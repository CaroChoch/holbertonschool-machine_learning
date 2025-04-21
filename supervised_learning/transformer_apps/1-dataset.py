#!/usr/bin/env python3
"""Encode a translation dataset into tokenized format"""
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Dataset class"""

    def __init__(self):
        """Class constructor"""
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )
        self.metadata = metadata
        self.data_train = examples['train']
        self.data_valid = examples['validation']

        # Build sub‑word tokenizers from the training corpus
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Replace raw datasets with tokenised versions
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Build two SubwordTextEncoder tokenisers from a tf.data.Dataset.
        The generator yields raw sentences so that SubwordTextEncoder
        can build its vocabulary without loading everything in memory.
        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.\
            build_from_corpus(
                (pt.numpy() for pt, _ in data),
                target_vocab_size=2 ** 15
            )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.\
            build_from_corpus(
                (en.numpy() for _, en in data),
                target_vocab_size=2 ** 15
            )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encode a pair of sentences (Portuguese, English) into lists of
        integer sub‑word indices, adding <start> and <end> tokens.
        """
        pt_start = self.tokenizer_pt.vocab_size
        en_start = self.tokenizer_en.vocab_size

        pt_tokens = [pt_start] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_start + 1]
        en_tokens = [en_start] + self.tokenizer_en.encode(
            en.numpy()) + [en_start + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper around self.encode so that it can be used
        inside tf.data pipelines.
        """
        pt_lang, en_lang = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_lang.set_shape([None])
        en_lang.set_shape([None])

        return pt_lang, en_lang
