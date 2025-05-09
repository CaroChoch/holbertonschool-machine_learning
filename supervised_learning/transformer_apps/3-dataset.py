#!/usr/bin/env python3
"""Set up the data pipeline"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """Dataset class"""

    def __init__(self, batch_size, max_len):
        """
        Class constructor:
        - batch_size: batch size for training/validation
        - max_len: maximum number of tokens allowed per sentence
        """
        # Load the TED talk translation dataset with info
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )
        self.data_train = examples['train']
        self.data_valid = examples['validation']
        self.metadata = metadata
        self.batch_size = batch_size
        self.max_len = max_len

        # Build tokenizers using a subset to avoid long runtimes
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Encode datasets using tf_encode wrapper
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        # Training data pipeline
        self.data_train = self.data_train.filter(self.filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            self.batch_size,
            padded_shapes=([None], [None])
        )
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        # Validation data pipeline
        self.data_valid = self.data_valid.filter(self.filter_max_len)
        self.data_valid = self.data_valid.padded_batch(
            self.batch_size,
            padded_shapes=([None], [None])
        )

    def tokenize_dataset(self, data):
        """
        Create SubwordTextEncoder tokenizers using first 10,000 examples
        to speed up tokenizer construction.
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in tfds.as_numpy(data.take(10000)):
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_sentences,
            target_vocab_size=2**15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_sentences,
            target_vocab_size=2**15
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encode a pair of sentences into token IDs with start/end tokens.
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
        TensorFlow wrapper around the encode method.
        Returns two tf.int64 tensors with shape [None].
        """
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens

    def filter_max_len(self, pt, en):
        """
        Filter out examples where pt or en exceeds max_len tokens.
        """
        return tf.logical_and(
            tf.size(pt) <= self.max_len,
            tf.size(en) <= self.max_len
        )
