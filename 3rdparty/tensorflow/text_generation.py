#!/usr/bin/env python
"""
The original version of file is downloaded from the below URL:
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/sequences/text_generation.ipynb

I made extensive modifications to structure the code the way I want as well as annotating.
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

import logging

import numpy as np

import tensorflow as tf

tf.enable_eager_execution()

FILE_PATH = '/tmp/shakespeare.txt'
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 1024

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
    char2idx = {u: i for i, u in enumerate(vocab)}  # e.g. char2idx['A'] = 13
    idx2char = np.array(vocab)  # idx2char[13] = 'A'

    # Tokenize: Convert the text to an array of integers
    text_as_int = np.array([char2idx[c] for c in text])  # [18 47 56 57 58  1 15 47 58 47 ...]

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

    # Create TensorFlow dataset
    char_tf_dataset_int_seq = tf.data.Dataset.from_tensor_slices(
        text_as_int)  # Feed # [18 47 56 57 58  1 15 47 58 47 ...]

    return char_tf_dataset_int_seq, char2idx, idx2char, vocab, seq_length, examples_per_epoch


def build_sequence():
    char_tf_dataset_int_seq, char2idx, idx2char, vocab, seq_length, examples_per_epoch = create_tf_dataset()

    # Output 5 characters
    for i in char_tf_dataset_int_seq.take(5):  # Equivalent of char_dataset[:5]
        print(idx2char[i.numpy()])  # convert to numpy format, then to a char.

    # Break the 1 million int sequence to batches with size 101. Drop the last portion that does not fill the batch size.
    sequences = char_tf_dataset_int_seq.batch(seq_length + 1, drop_remainder=True)

    for item in sequences.take(5):  # Take the initial 5 batches
        print(repr(''.join(idx2char[item.numpy()])))  # Concatenate characters after converting from int.

    return sequences, char2idx, idx2char, vocab, examples_per_epoch


def split_input_target(chunk):
    input_text = chunk[:-1]  # everything but the last letter
    target_text = chunk[1:]  # everything but the first letter
    return input_text, target_text


def load_data():
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

    # Print out:
    #   First input data
    #   First target data
    for input_example, target_example in dataset.take(1):
        print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

        # Print out:
        #  First 5 letters of input data
        #  First 5 letters of target data
        for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
            print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

    steps_per_epoch = examples_per_epoch // BATCH_SIZE  # e.g. steps per epoch = 174

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    return dataset, char2idx, idx2char, vocab, vocab_size, steps_per_epoch


def setup_rnn_layer():
    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU
    else:
        import functools

        # Define rnn as GRU(ecurrent_activation='sigmoid')
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
    return model


def loss(labels, logits):
    # return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def generate_text(model, char2idx, idx2char, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_characters_to_generate = 1000

    # You can change the start string to experiment
    start_string = 'ROMEO'

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    next_input = tf.expand_dims(input_eval, 0)  # Equivalent of input_eval = input_eval.reshape([1, input_eval.shape[0])

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_characters_to_generate):
        predictions = model(next_input)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        next_input = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


def main():
    dataset, char2idx, idx2char, vocab, vocab_size, steps_per_epoch = load_data()
    rnn_layer = setup_rnn_layer()

    model = build_model(rnn_layer,
                        vocab_size=vocab_size,
                        batch_size=BATCH_SIZE)

    example_batch_predictions = None  # HI
    input_example_batch = None  # HI
    target_example_batch = None  # HI

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.summary()

    # sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.random.multinomial(example_batch_predictions[0], num_samples=1)

    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS = 3

    history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
                        callbacks=[checkpoint_callback])
    print("Training completed.")

    # Predict
    tf.train.latest_checkpoint(checkpoint_dir)
    model = build_model(rnn_layer, vocab_size, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    model.summary()

    print(generate_text(model, char2idx, idx2char, start_string="ROMEO: "))


if __name__ == "__main__":
    main()
