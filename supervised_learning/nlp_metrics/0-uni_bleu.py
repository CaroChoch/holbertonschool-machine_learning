#!/usr/bin/env python3
""" calculates the unigram BLEU score for a sentence """
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    Arguments:
        - references: list of reference translations
            * each reference translation is a list of the words
            in the translation
        - sentence: list containing the model proposed sentence
    Returns:
        - The unigram BLEU score
    """
    # Step 1: Calculate the length of the proposed sentence
    sentence_len = len(sentence)

    # Step 2: Find the reference length closest to the proposed sentence length
    ref_len = []
    for ref in references:
        ref_len.append(len(ref))
    ref_len = np.array(ref_len)
    closest_ref_idx = np.argmin(np.abs(ref_len - sentence_len))
    closest_ref_len = len(references[closest_ref_idx])

    # Step 3: Calculate unigram precision
    word_counts = {}
    for word in sentence:
        for ref in references:
            if word in ref:
                if word not in word_counts:
                    word_counts[word] = ref.count(word)
                else:
                    word_counts[word] = max(word_counts[word], ref.count(word))

    precision = sum(word_counts.values()) / sentence_len

    # Step 4: Calculate brevity penalty
    if sentence_len > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_len)

    # Step 5: Calculate BLEU score
    bleu_score = brevity_penalty * precision

    return bleu_score
