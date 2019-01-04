#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The original version of file was downloaded from the below URL:
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/sequences/text_generation.ipynb

I made extensive modifications to:
 * use Japanese books as data source
 * structure the code the way I want
 * add annotations.
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
import sys

import logging
from pathlib import Path
import numpy as np

import tensorflow as tf

BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 1024
EPOCHS = 100
START_STRING = '私'
DEFAULT_START_STRING = "それ"  # if START_STRING is not found in text, this string is used.
INPUT_DIR = '/tmp/ml_examples/japanese_books'
CHECKPOINT_DIR = '/tmp/ml_examples/training_checkpoints_text_generation_japanese'

tf.enable_eager_execution()

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def create_tf_dataset():
    """
    Load Shakespeare data.
    """

    input_dir_path = Path(INPUT_DIR)
    if input_dir_path.exists() is False:
        log.fatal("% does not exist." % (input_dir_path))
        sys.exit(1)

    original_books = ""
    # Process files
    for input_file in input_dir_path.glob("*.txt"):
        with open(input_file, "r") as f:
            original_book = f.read()  # Change this if you are reading a very large text file.
            original_books += (original_book + "\n")

    text_to_process = original_books
    # The unique letters in the file
    vocab = sorted(set(text_to_process))
    print('{} unique letters'.format(len(vocab)))

    # Creating a mapping from unique letters to indices
    letter2idx = {w: i for i, w in enumerate(vocab)}  # e.g. letter2idx['A'] = 13
    idx2letter = np.array(vocab)  # idx2letter[13] = 'A'

    # Tokenize: Convert the text to an array of integers
    text_as_int = np.array([letter2idx[w] for w in text_to_process])  # [18 47 56 57 58  1 15 47 58 47 ...]

    # Show the first 20 letters and IDs
    print('{')
    for letter, _ in zip(letter2idx, range(20)):
        print('  %s: %3d,' % (letter, letter2idx[letter]))
    print('  ...\n}')

    # Show how the first 13 letters from the text are mapped to integers
    print('{} ---- letters mapped to int ---- > {}'.format(repr(text_to_process[:13]), text_as_int[:13]))

    # The maximum length sentence we want for a single input in letters
    seq_length = 100
    examples_per_epoch = len(text_to_process) // seq_length

    # Create TensorFlow dataset
    letter_tf_dataset_int_seq = tf.data.Dataset.from_tensor_slices(
        text_as_int)  # Feed # [18 47 56 57 58  1 15 47 58 47 ...]

    return letter_tf_dataset_int_seq, letter2idx, idx2letter, vocab, seq_length, examples_per_epoch


def build_sequence():
    letter_tf_dataset_int_seq, letter2idx, idx2letter, vocab, seq_length, examples_per_epoch = create_tf_dataset()

    # Output 5 letters
    for i in letter_tf_dataset_int_seq.take(5):  # Equivalent of char_dataset[:5]
        print(idx2letter[i.numpy()])  # convert to numpy format, then to a char.

    # Break the 1 million int sequence to batches with size 101. Drop the last portion that does not fill the batch size.
    sequences = letter_tf_dataset_int_seq.batch(seq_length + 1, drop_remainder=True)

    for item in sequences.take(5):  # Take the initial 5 batches
        print(repr(''.join(idx2letter[item.numpy()])))  # Concatenate letters after converting from int.

    return sequences, letter2idx, idx2letter, vocab, examples_per_epoch


def split_input_target(chunk):
    x = chunk[:-1]  # everything but the last letter
    y = chunk[1:]  # everything but the first letter
    return x, y


def load_data():
    # Build batches of int arrays of size 101 representing letters
    sequences, letter2idx, idx2letter, vocab, examples_per_epoch = build_sequence()

    # For a string: "Apples are delicious."
    # Create two sets of data.
    # 1) Apples are delicious (without the last symbol which is a period)
    # 2) pples are delicious (without the first letter)
    # Use 1) as input and 2) as output
    # The goal is to predict each element of 2) from each element of 1).
    # Namely, when 'A' is input from 1), then try to guess 'p' from 2).
    # when 'p' is input from 1), try to guess 'p' from 2)
    dataset = sequences.map(split_input_target)

    # Print out:
    #   First input data
    #   First target data
    for input_example, target_example in dataset.take(1):
        print('Input data: ', repr(''.join(idx2letter[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2letter[target_example.numpy()])))

        # Print out:
        #  First 5 letters of input data
        #  First 5 letters of target data
        for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(input_idx, repr(idx2letter[input_idx])))
            print("  expected output: {} ({:s})".format(target_idx, repr(idx2letter[target_idx])))

    steps_per_epoch = examples_per_epoch // BATCH_SIZE  # e.g. steps per epoch = 174

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    return dataset, letter2idx, idx2letter, vocab, vocab_size, steps_per_epoch


def setup_rnn_layer():
    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU
    else:
        import functools

        # Define rnn as GRU(current_activation='sigmoid')
        rnn = functools.partial(
            tf.keras.layers.GRU, recurrent_activation='sigmoid')

    return rnn


def build_model(rnn_layer, vocab_size, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM,
                                  batch_input_shape=[batch_size, None]),
        rnn_layer(RNN_UNITS,
                  return_sequences=True,
                  recurrent_initializer='glorot_uniform',
                  stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])

    model.summary()

    return model


def loss(labels, logits):
    # return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def generate_text(model, letter2idx, idx2letter, start_string=START_STRING):
    # Evaluation step (generating text using the learned model)

    # Number of letters to generate
    num_letters_to_generate = 1000

    # Converting our start string to numbers (vectorizing)
    if start_string not in letter2idx:
        start_string = DEFAULT_START_STRING

    input_eval = [letter2idx[s] for s in start_string]
    next_input = tf.expand_dims(input_eval, 0)  # Equivalent of input_eval = input_eval.reshape([1, input_eval.shape[0])

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_letters_to_generate):
        predictions = model(next_input)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the letter returned by the model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted letter as the next input to the model
        # along with the previous hidden state
        next_input = tf.expand_dims([predicted_id], 0)

        letter = idx2letter[predicted_id]
        text_generated.append(letter)

    return (start_string + ''.join(text_generated))


def show_untrained_prediction(dataset, model, idx2letter):
    example_batch_predictions = None
    input_example_batch = None
    target_example_batch = None

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    # sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.random.multinomial(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    print("Input: \n", repr("".join(idx2letter[input_example_batch[0]])))
    print()
    print("Next Word Predictions: \n", repr("".join(idx2letter[sampled_indices])))

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())


def setup_checkpoint_callback():
    checkpoint_dir = Path(CHECKPOINT_DIR)  # Directory where the checkpoints will be saved

    if checkpoint_dir.exists() is False:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_prefix = checkpoint_dir / Path("ckpt_{epoch}")  # Name of the checkpoint files
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_prefix),
        save_weights_only=True)
    return checkpoint_callback, str(checkpoint_dir)


def main():
    dataset, letter2idx, idx2letter, vocab, vocab_size, steps_per_epoch = load_data()
    rnn_layer = setup_rnn_layer()

    model = build_model(rnn_layer,
                        vocab_size=vocab_size,
                        batch_size=BATCH_SIZE)

    show_untrained_prediction(dataset, model, idx2letter)

    model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

    checkpoint_callback, checkpoint_dir = setup_checkpoint_callback()

    if Path(checkpoint_dir).exists():
        tf.train.latest_checkpoint(checkpoint_dir)
        try:
            model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
            log.info("Loaded weight")
        except AttributeError:
            log.info("Weight not found.  Proceeding")
    else:
        log.info("Weight not found.  Proceeding")

    # 174 batches/step in each epoch. Each batch has 64 samples.
    # Each sample contains a pair of x & y each of which has 100 letters
    history = model.fit(dataset.repeat(),  # Repeat dataset indefinitely
                        epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[checkpoint_callback])
    print("Training completed.")

    # Predict
    tf.train.latest_checkpoint(checkpoint_dir)
    model = build_model(rnn_layer, vocab_size, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    print(generate_text(model, letter2idx, idx2letter))


if __name__ == "__main__":
    main()
