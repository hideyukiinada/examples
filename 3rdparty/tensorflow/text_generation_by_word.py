#!/usr/bin/env python
"""
The original version of file was downloaded from the below URL:
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/sequences/text_generation.ipynb

I made extensive modifications to:
 * predict a word instead of a character
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

import logging
from pathlib import Path
import numpy as np

import tensorflow as tf
import keras.preprocessing.text

BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 1024
EPOCHS = 10
START_STRING = 'ROMEO:'
DEFAULT_START_STRING = "Romeo" # if START_STRING is not found in text, this string is used.
URL = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
FILE_PATH = '/tmp/shakespeare.txt'
CHECKPOINT_DIR = '/tmp/ml_examples/training_checkpoints_text_generation_by_word'
EOS_MARKER = "<EOS>"

tf.enable_eager_execution()

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def create_tf_dataset():
    """
    Load Shakespeare data.
    """
    path_to_file = tf.keras.utils.get_file(FILE_PATH, URL)

    original_text = open(path_to_file).read()

    # Process end of sentence
    updated_text = original_text.replace(". ", EOS_MARKER)

    filters = '\'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n'
    words_in_text = keras.preprocessing.text.text_to_word_sequence(updated_text,
                                                                   filters=filters,lower=False, split=' ')

    # The unique words in the file
    vocab = sorted(set(words_in_text))
    print('{} unique words'.format(len(vocab)))

    # Creating a mapping from unique words to indices
    word2idx= {w: i for i, w in enumerate(vocab)}  # e.g. word2idx['A'] = 13
    idx2word = np.array(vocab)  # idx2word[13] = 'A'

    # Tokenize: Convert the text to an array of integers
    text_as_int = np.array([word2idx[w] for w in words_in_text])  # [18 47 56 57 58  1 15 47 58 47 ...]

    # Show the first 20 words and IDs
    print('{')
    for word, _ in zip(word2idx, range(20)):
        print('  %s: %3d,' % (word, word2idx[word]))
    print('  ...\n}')

    # Show how the first 13 words from the text are mapped to integers
    print('{} ---- words mapped to int ---- > {}'.format(repr(words_in_text[:13]), text_as_int[:13]))

    # The maximum length sentence we want for a single input in words
    seq_length = 100
    examples_per_epoch = len(words_in_text) // seq_length

    # Create TensorFlow dataset
    word_tf_dataset_int_seq = tf.data.Dataset.from_tensor_slices(
        text_as_int)  # Feed # [18 47 56 57 58  1 15 47 58 47 ...]

    return word_tf_dataset_int_seq, word2idx, idx2word, vocab, seq_length, examples_per_epoch


def build_sequence():
    word_tf_dataset_int_seq, word2idx, idx2word, vocab, seq_length, examples_per_epoch = create_tf_dataset()

    # Output 5 words
    for i in word_tf_dataset_int_seq.take(5):  # Equivalent of char_dataset[:5]
        print(idx2word[i.numpy()])  # convert to numpy format, then to a char.

    # Break the 1 million int sequence to batches with size 101. Drop the last portion that does not fill the batch size.
    sequences = word_tf_dataset_int_seq.batch(seq_length + 1, drop_remainder=True)

    for item in sequences.take(5):  # Take the initial 5 batches
        print(repr(''.join(idx2word[item.numpy()])))  # Concatenate words after converting from int.

    return sequences, word2idx, idx2word, vocab, examples_per_epoch


def split_input_target(chunk):
    x = chunk[:-1]  # everything but the last letter
    y = chunk[1:]  # everything but the first letter
    return x, y


def load_data():
    # Build batches of int arrays of size 101 representing letters
    sequences, word2idx, idx2word, vocab, examples_per_epoch = build_sequence()

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
        print('Input data: ', repr(''.join(idx2word[input_example.numpy()])))
        print('Target data:', repr(''.join(idx2word[target_example.numpy()])))

        # Print out:
        #  First 5 letters of input data
        #  First 5 letters of target data
        for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(input_idx, repr(idx2word[input_idx])))
            print("  expected output: {} ({:s})".format(target_idx, repr(idx2word[target_idx])))

    steps_per_epoch = examples_per_epoch // BATCH_SIZE  # e.g. steps per epoch = 174

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    return dataset, word2idx, idx2word, vocab, vocab_size, steps_per_epoch


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


def generate_text(model, word2idx, idx2word, start_string=START_STRING):
    # Evaluation step (generating text using the learned model)

    # Number of words to generate
    num_words_to_generate = 1000

    # Converting our start string to numbers (vectorizing)
    if start_string not in word2idx:
        start_string = DEFAULT_START_STRING

    input_eval = [word2idx[start_string]]
    next_input = tf.expand_dims(input_eval, 0)  # Equivalent of input_eval = input_eval.reshape([1, input_eval.shape[0])

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_words_to_generate):
        predictions = model(next_input)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        next_input = tf.expand_dims([predicted_id], 0)

        word = idx2word[predicted_id]
        if word == EOS_MARKER:
            word = ".\n"
        else:
            word += " "
        text_generated.append(word)

    return (start_string + ''.join(text_generated))


def show_untrained_prediction(dataset, model, idx2word):
    example_batch_predictions = None
    input_example_batch = None
    target_example_batch = None

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    # sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.random.multinomial(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    print("Input: \n", repr("".join(idx2word[input_example_batch[0]])))
    print()
    print("Next Word Predictions: \n", repr("".join(idx2word[sampled_indices])))

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
    dataset, word2idx, idx2word, vocab, vocab_size, steps_per_epoch = load_data()
    rnn_layer = setup_rnn_layer()

    model = build_model(rnn_layer,
                        vocab_size=vocab_size,
                        batch_size=BATCH_SIZE)

    show_untrained_prediction(dataset, model, idx2word)

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
    # Each sample contains a pair of x & y each of which has 100 words
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

    print(generate_text(model, word2idx, idx2word))


if __name__ == "__main__":
    main()
