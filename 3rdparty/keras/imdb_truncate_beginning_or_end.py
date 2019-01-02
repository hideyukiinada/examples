#!/usr/bin/env python
"""
The original version of file is downloaded from the below URL:
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb

I made minor modifications.
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

TRAINING_DATA_SET_SIZE = 10000
NUM_WORDS_TO_LOAD = 10000
NUM_WORDS_PER_REVIEW = 256

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def decode_review(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def load_data(truncating_option):
    """
    Load IMDB data

    Parameters
    ----------
    truncating_option: str
        Controls whether to truncate at the beginning or the end

    Returns
    -------
    (x_train, y_train), (x_val, y_val), (x_test, y_test): tuple of ndarray
        x_train: training dataset,
        y_train: training dataset ground truth
        x_val: validation dataset,
        y_val: validation dataset ground truth
        x_test: test dataset,
        y_test: test dataset ground truth
    """

    imdb = keras.datasets.imdb

    (x_train_raw, y_train), (x_test_raw, y_test) = imdb.load_data(num_words=NUM_WORDS_TO_LOAD)
    log.info("Training entries: {}, labels: {}".format(len(x_train_raw), len(y_train)))

    log.info(x_train_raw[0])

    len(x_train_raw[0]), len(x_train_raw[1])

    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    decode_review(x_train_raw[0], reverse_word_index)

    x_train = keras.preprocessing.sequence.pad_sequences(x_train_raw,
                                                         value=word_index["<PAD>"],
                                                         padding='post',
                                                         maxlen=NUM_WORDS_PER_REVIEW, truncating=truncating_option)

    x_test = keras.preprocessing.sequence.pad_sequences(x_test_raw,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=NUM_WORDS_PER_REVIEW, truncating=truncating_option)

    len(x_train[0]), len(x_train[1])

    log.info(x_train[0])

    x_val = x_train[:TRAINING_DATA_SET_SIZE]
    partial_x_train = x_train[TRAINING_DATA_SET_SIZE:]

    y_val = y_train[:TRAINING_DATA_SET_SIZE]
    partial_y_train = y_train[TRAINING_DATA_SET_SIZE:]

    return (partial_x_train, partial_y_train), (x_val, y_val), (x_test, y_test)


def train(x_train, y_train, x_val, y_val, use_global_average_pooling=False):
    """
    Train the model

    Parameters
    ----------
    x_train: ndarray
        training dataset
    y_train: ndarray
        training dataset ground truth
    x_val: ndarray
        validation dataset
    y_val: ndarray
        validation dataset ground truth
    use_global_average_pooling: bool
        If true, calculate the mean of the embedding of words for the entire sentence instead of retaining
        the embedding for each word

    Returns
    -------
    model: Keras.model.Sequential
        model
    history: History
        History of loss values and metrics.  For further details, see https://keras.io/models/sequential/
    """

    # input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = NUM_WORDS_TO_LOAD

    model = Sequential()

    if use_global_average_pooling:
        model.add(keras.layers.Embedding(vocab_size, 16))
        model.add(keras.layers.GlobalAveragePooling1D())
    else:
        model.add(Embedding(vocab_size, 16, input_length=NUM_WORDS_PER_REVIEW))
        model.add(Flatten())

    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])

    history = model.fit(x_train,
                        y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    return model, history


def plot(history, label):
    """
    Plot stats using matplotlib

    Parameters
    ----------
    history: History
        History of loss values and metrics.  For further details, see https://keras.io/models/sequential/
    label: str
        Label to show on top of the plot
    """

    history_dict = history.history
    history_dict.keys()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss (%s)' % (label))
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss (%s)' % (label))
    plt.title('Training and validation loss (%s)' % (label))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()  # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc (%s)' % (label))
    plt.plot(epochs, val_acc, 'b', label='Validation acc (%s)' % (label))
    plt.title('Training and validation accuracy (%s)' % (label))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def main():
    """
    Load data, train and evaluate the IMDB data.
    """
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    history_list = list()
    plot_label_list = list()

    for use_average_pooling in (True, False):
        model, history = train(x_train, y_train, x_val, y_val, use_average_pooling)

        results = model.evaluate(x_test, y_test)
        log.info(results)

        history_list.append(history)
        plot_label_list.append("Ave pooling: " + str(use_average_pooling))

    for i, h in enumerate(history_list):
        plot(h, plot_label_list[i])


if __name__ == "__main__":
    main()
