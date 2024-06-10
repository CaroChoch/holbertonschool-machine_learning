#!/usr/bin/env python3
""" answers questions from a reference text """
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

exit_commands = ['exit', 'quit', 'goodbye', 'bye']


def answer_loop(reference):
    """
    Run the loop to answer questions
    Arguments:
        - reference: (str) containing the reference document from which to find
          the answer
    Returns: the answer or if the answer cannot be found in the reference text,
        respond 'with Sorry, I do not understand your question'
    """
    while True:
        # Prompt user for a question and remove any trailing newline characters
        question = input("Q: ").rstrip('\n')

        # Check if the user input is an exit command
        if question.lower() in exit_commands:
            print("A: Goodbye")
            break
        else:
            answer = question_answer(question, reference)
            if answer is None:
                answer = "Sorry, I do not understand your question."
            print(f"A: {answer}")


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Arguments:
        - question: (str) containing the question to answer
        - reference: (str) containing the reference document from which to find
          the answer

    Returns:
        (str) containing the answer
    """
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )

    # Load the pre-trained BERT model from TensorFlow Hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize the question and reference text
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    # Prepare the tokens for the BERT model input
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + \
        reference_tokens + ['[SEP]']

    # Convert tokens to input IDs
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Create input mask (1 for real tokens and 0 for padding tokens)
    input_mask = [1] * len(input_word_ids)

    # Create input type IDs (0 for question tokens, 1 for reference tokens)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + \
        [1] * (len(reference_tokens) + 1)

    # Convert lists to tensors and expand dimensions for batch compatibility
    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids)
    )

    # Get model outputs for start and end positions of the answer
    outputs = model([input_word_ids, input_mask, input_type_ids])

    # Find the start and end positions of the answer in the reference text
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    # Extract answer tokens from the original tokens
    answer_tokens = tokens[short_start: short_end + 1]

    # Convert answer tokens back to a string
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    # Return the answer if tokens are found, otherwise return None
    if not answer_tokens:
        return None

    return answer
