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

    sequences = char_tf_dataset_int_seq.batch(seq_length + 1, drop_remainder=True)

    for item in sequences.take(5):
        print(repr(''.join(idx2char[item.numpy()])))

    return sequences, idx2char, vocab, examples_per_epoch



def main():
    sequences, idx2char, vocab, examples_per_epoch = build_sequence()


if __name__ == "__main__":
    main()
