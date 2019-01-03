#!/usr/bin/env python
"""
The original version of file is downloaded from the below URL:
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/sequences/text_generation.ipynb

I made minor modifications to print reviews only.
"""
# Copyright 2018 The TensorFlow Authors.
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import logging

import numpy as np

import tensorflow as tf

tf.enable_eager_execution()

FILE_PATH = '/tmp/shakespeare.txt'

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def create_tf_dataset():
    """
    Load Shakespeare data.
    """
    path_to_file = tf.keras.utils.get_file(FILE_PATH,
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    text = open(path_to_file).read()

    # length of text is the number of characters in it
    print('Length of text: {} characters'.format(len(text)))

    # Take a look at the first 250 characters in text
    print(text[:250])

    # The unique characters in the file
    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))

    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)} # e.g. char2idx['A'] = 13
    idx2char = np.array(vocab) # idx2char[13] = 'A'

    # Convert the text to an array of integers
    text_as_int = np.array([char2idx[c] for c in text]) # [18 47 56 57 58  1 15 47 58 47 ...]

    # Show the first 20 characters and IDs
    print('{')
    for char, _ in zip(char2idx, range(20)):
        print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    print('  ...\n}')

    # Show how the first 13 characters from the text are mapped to integers
    print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

    # The maximum length sentence we want for a single input in characters
    seq_length = 100
    examples_per_epoch = len(text) // seq_length  # Length of text: 1115394 characters. // 100 = 11153

    # Create training examples / targets
    char_tf_dataset_int_seq = tf.data.Dataset.from_tensor_slices(text_as_int) # Feed # [18 47 56 57 58  1 15 47 58 47 ...]

    return char_tf_dataset_int_seq, char2idx, idx2char, vocab, seq_length, examples_per_epoch

def build_sequence():
    char_tf_dataset_int_seq, char2idx, idx2char, vocab, seq_length, examples_per_epoch = create_tf_dataset()

    # Output 5 characters
    for i in char_tf_dataset_int_seq.take(5): # Equivalent of char_dataset[:5]
        print(idx2char[i.numpy()])  # convert to numpy format, then to a char.

    # Break the 1 million int sequence to batches with size 101. Drop the last portion that does not fill the batch size.
    sequences = char_tf_dataset_int_seq.batch(seq_length + 1, drop_remainder=True)

    for item in sequences.take(5):  # Take the initial 5 batches
        print(repr(''.join(idx2char[item.numpy()])))  # Concatenate characters after converting from int.

    return sequences, char2idx, idx2char, vocab, examples_per_epoch


def split_input_target(chunk):
    input_text = chunk[:-1] # everything but the last letter
    target_text = chunk[1:] # everything but the first letter
    return input_text, target_text


def setup_rnn():
    # Build batches of int arrays of size 101 representing letters
    sequences, char2idx, idx2char, vocab, examples_per_epoch = build_sequence()

    # For a string: "Apples are delicious."
    # Create two sets of data.
    # 1) Apples are delicious (without the last symbol which is a period)
    # 2) pples are delicious (without the first letter)
    # Use 1) as input and 2) as output
    # The goal is to predict each element of 2) from each element of 1).
    # Namely, when 'A' is input from 1), then try to guess 'p' from 2).
    # when 'p' is input from 1), try to guess 'p' from 2)
    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

        for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
            print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

    # Batch size
    BATCH_SIZE = 64
    steps_per_epoch = examples_per_epoch // BATCH_SIZE

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU
    else:
        import functools

        rnn = functools.partial(
            tf.keras.layers.GRU, recurrent_activation='sigmoid')

    return rnn_units, rnn, embedding_dim

def main():
    setup_rnn()


if __name__ == "__main__":
    main()
