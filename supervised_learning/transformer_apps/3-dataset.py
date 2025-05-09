#!/usr/bin/env python3
"""
Prepare, encode, and set up a data pipeline for the TED HRLR
Portuguese-English translation dataset using pre-trained tokenizers.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Load and preprocess the TED HRLR translation dataset."""

    def __init__(self, batch_size, max_len):
        """Initialize dataset, tokenizers, and data pipeline."""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.data_train = self.data_train.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        self.data_train = self.data_train.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len))
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(buffer_size=20000)
        self.data_train = self.data_train.padded_batch(
            batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(
            buffer_size=tf.data.AUTOTUNE)

        self.data_valid = self.data_valid.map(
            self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        self.data_valid = self.data_valid.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len))
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self, data):
        """Create and train tokenizers from the dataset."""
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True, clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode sentences and add start/end tokens."""
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence, add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(
            en_sentence, add_special_tokens=False)

        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + \
            [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + \
            [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for encode method."""
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64])

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
