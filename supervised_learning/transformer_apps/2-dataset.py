#!/usr/bin/env python3
"""TF encode method"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """Dataset class"""

    def __init__(self):
        """Class constructor"""
        examples, metadata = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            with_info=True,
            as_supervised=True
        )
        self.metadata = metadata
        self.data_train = examples["train"]
        self.data_valid = examples["validation"]

        # Build sub-word tokenizers from the training corpus
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Map datasets to tokenized versions
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Builds sub-word tokenizers for pt and en"""
        pt_texts = [pt.numpy().decode('utf-8') for pt, _ in tfds.as_numpy(data)]
        en_texts = [en.numpy().decode('utf-8') for _, en in tfds.as_numpy(data)]

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            pt_texts, target_vocab_size=2 ** 15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            en_texts, target_vocab_size=2 ** 15
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes pt and en sentences to token IDs with start/end tokens"""
        pt_start = self.tokenizer_pt.vocab_size
        en_start = self.tokenizer_en.vocab_size

        # Decode tensor bytes into strings
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = [pt_start] + self.tokenizer_pt.encode(pt_text) + [pt_start + 1]
        en_tokens = [en_start] + self.tokenizer_en.encode(en_text) + [en_start + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper around self.encode for use in dataset.map()"""
        pt_lang, en_lang = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_lang.set_shape([None])
        en_lang.set_shape([None])

        return pt_lang, en_lang
