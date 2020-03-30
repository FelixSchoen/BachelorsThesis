import tensorflow as tf
import numpy as np
import os
import time


class Test:

    @staticmethod
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    @staticmethod
    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(rnn_units,
                                 return_sequences=True,
                                 stateful=True,
                                 recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    @staticmethod
    def test():
        path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                               'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

        # Read, then decode for py2 compat.
        text = open(path_to_file, 'rb').read().decode(encoding="utf-8")

        vocab = sorted(set(text))
        print('{} unique characters'.format(len(vocab)))

        # Creating a mapping from unique characters to indices
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        text_as_int = np.array([char2idx[c] for c in text])

        # The maximum length sentence we want for a single input in characters
        seq_length = 100
        examples_per_epoch = len(text) // (seq_length + 1)

        # Create training examples / targets
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

        for i in char_dataset.take(5):
            print(idx2char[i])

        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

        for item in sequences.take(5):
            print(repr(''.join(idx2char[item.numpy()])))

        dataset = sequences.map(Test.split_input_target)

        for input_example, target_example in dataset.take(1):
            print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
            print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

        for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
            print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

        # Batch size
        BATCH_SIZE = 64

        # Buffer size to shuffle the dataset
        # (TF data is designed to work with possibly infinite sequences,
        # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        # it maintains a buffer in which it shuffles elements).
        BUFFER_SIZE = 10000

        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        # Batch size
        BATCH_SIZE = 64

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

        model = Test.build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)

        print(model.summary())


if __name__ == '__main__':
    Test.test()
