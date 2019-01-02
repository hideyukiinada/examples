#!/usr/bin/env python
"""
The original version of file is downloaded from the below URL:
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb

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
# @title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import os
import logging

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten

import matplotlib.pyplot as plt

NUM_WORDS_TO_LOAD = 10000
NUM_WORDS_PER_REVIEW = 256
NUM_REVIEWS_TO_PRINT = 20

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def decode_review(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def load_data():
    """
    Load IMDB data
    """

    imdb = keras.datasets.imdb

    (x_train_raw, y_train), (x_test_raw, y_test) = imdb.load_data(num_words=NUM_WORDS_TO_LOAD)
    log.info("Training entries: {}, labels: {}".format(len(x_train_raw), len(y_train)))

    log.info(x_train_raw[0])

    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    for i in range(NUM_REVIEWS_TO_PRINT):
        decoded_review = decode_review(x_train_raw[i], reverse_word_index)
        print("[%d][word count:%d] %s" % (i, len(x_train_raw[i]), decoded_review))

    x_train = keras.preprocessing.sequence.pad_sequences(x_train_raw,
                                                         value=word_index["<PAD>"],
                                                         padding='post',
                                                         maxlen=NUM_WORDS_PER_REVIEW)

    for i in range(NUM_REVIEWS_TO_PRINT):
        decoded_review = decode_review(x_train[i], reverse_word_index)
        print("[%d][word count:%d] %s" % (i, len(x_train[i]), decoded_review))


def main():
    """
    Load IMDB data.
    """
    load_data()

if __name__ == "__main__":
    main()
