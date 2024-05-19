#!/usr/bin/env python3
""" Calculates the cumulative n-gram BLEU score for a sentence """
import numpy as np
from collections import Counter


def generate_ngrams(sentence, n):
    """Generate n-grams for a given sentence."""
    slices = []
    for i in range(n):
        slices.append(sentence[i:])
    return Counter(zip(*slices))


def generate_ref_ngrams(references, n):
    """Generate n-grams for references."""
    reference_ngrams = []
    for ref in references:
        reference_ngrams.append(generate_ngrams(ref, n))
    return reference_ngrams


def calculate_clipped_counts(sentence_ngrams, reference_ngrams):
    """Calculate clipped counts for a given sentence and references."""
    clipped_counts = {}
    for ngram, count in sentence_ngrams.items():
        max_ref_count = 0
        for ref_ngram in reference_ngrams:
            if ngram in ref_ngram:
                max_ref_count = max(max_ref_count, ref_ngram[ngram])
        clipped_counts[ngram] = min(count, max_ref_count)
    return clipped_counts


def calculate_precision(clipped_counts, sentence_ngrams):
    """Calculate precision."""
    return sum(clipped_counts.values()) / max(1, sum(sentence_ngrams.values()))


def calculate_brevity_penalty(references, sentence):
    """Calculate brevity penalty."""
    reference_lengths = [len(ref) for ref in references]
    len_sentence = len(sentence)
    closest_ref_length = min(
        reference_lengths,
        key=lambda ref_len: (abs(ref_len - len_sentence), ref_len))
    if len_sentence < closest_ref_length:
        return np.exp(1 - closest_ref_length / len_sentence)
    return 1


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence
    Arguments:
        - references is a list of reference translations
            * each reference translation is a list of the words in the
            translation
        - sentence is a list containing the model proposed sentence
        - n is the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score
    """
    sentence_ngrams = generate_ngrams(sentence, n)
    reference_ngrams = generate_ref_ngrams(references, n)
    clipped_counts = calculate_clipped_counts(
        sentence_ngrams, reference_ngrams)
    precision = calculate_precision(clipped_counts, sentence_ngrams)
    brevity_penalty = calculate_brevity_penalty(references, sentence)
    return brevity_penalty * precision


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence
    Arguments:
        - references is a list of reference translations
            * each reference translation is a list of the words in the
            translation
        - sentence is a list containing the model proposed sentence
        - n is the size of the largest n-gram to use for evaluation
    Returns: the cumulative n-gram BLEU score
    """
    # Calculating the BLEU score for each n-gram
    bleu_scores = []
    for i in range(1, n + 1):
        bleu_scores.append(ngram_bleu(references, sentence, i))

    # Calculating the weights for each n-gram
    weights = [1 / n for i in range(n)]

    # Calculating the cumulative BLEU score
    cumulative_bleu_score = np.exp(np.sum(weights * np.log(bleu_scores)))

    return cumulative_bleu_score
