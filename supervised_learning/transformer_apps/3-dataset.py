#!/usr/bin/env python3
"""Set up the data pipeline"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """Dataset class"""

    def __init__(self, batch_size, max_len):
        """
        Class constructor
        Arguments:
            - batch_size: integer, batch size for training/validation
            - max_len: integer, max number of tokens allowed per sentence
        """
        # Load dataset with info and as_supervised
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

        # Build subword tokenizers from train subset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Encode datasets
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        # Training pipeline: filter, cache, shuffle, batch, prefetch
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

        # Validation pipeline: filter, batch
        self.data_valid = self.data_valid.filter(self.filter_max_len)
        self.data_valid = self.data_valid.padded_batch(
            self.batch_size,
            padded_shapes=([None], [None])
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for dataset
        using pretrained models trained on our data
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in tfds.as_numpy(data):
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        pretrained_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True
        )
        pretrained_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True
        )

        tokenizer_pt = pretrained_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2**13
        )
        tokenizer_en = pretrained_en.train_new_from_iterator(
            en_sentences, vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens with start/end markers
        """
        pt_start = self.tokenizer_pt.vocab_size
        pt_end = pt_start + 1
        en_start = self.tokenizer_en.vocab_size
        en_end = en_start + 1

        pt_tokens = [pt_start] + self.tokenizer_pt.encode(pt.numpy()) + [pt_end]
        en_tokens = [en_start] + self.tokenizer_en.encode(en.numpy()) + [en_end]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper around the encode method
        Returns two tf.int64 tensors shaped [None]
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
        Filters out examples longer than max_len tokens
        """
        return tf.logical_and(
            tf.size(pt) <= self.max_len,
            tf.size(en) <= self.max_len
        )
