from __future__ import print_function

import keras as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from src.Utility import *
from src.MusicElements import *
import numpy as np
import os

VOCAB_SIZE = 203


def shizzle():
    batch_size = 64  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    num_samples = 10000  # Number of samples to train on.
    # Path to the data txt file on disk.
    data_path = 'fra-eng/fra.txt'

    treble_sequences, bass_sequences, target = load_pickle_data(Complexity.MEDIUM, batch_size)

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    x = K.layers.Embedding(VOCAB_SIZE, latent_dim)(encoder_inputs)
    x, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    x = K.layers.Embedding(VOCAB_SIZE, latent_dim)(decoder_inputs)
    x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
    decoder_outputs = Dense(VOCAB_SIZE, activation="softmax")(x)
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit([treble_sequences, bass_sequences], target,
              batch_size=128,
              epochs=epochs,
              validation_split=0.2)
    # Save model
    model.save('s2s.h5')


def split(chunk):
    seq_input = chunk[:-1]
    seq_output = chunk[1:]
    return seq_input, seq_output


def load_pickle_data(complexity, batch_size):
    if complexity == Complexity.EASY:
        path = "../../out/lib/4-4/easy"
    elif complexity == Complexity.MEDIUM:
        path = "../../out/lib/4-4/medium"
    else:
        path = "../../out/lib/4-4/hard"

    treble_sequences = []
    bass_sequences = []

    for (dirpath, dirnames, filenames) in os.walk(path):
        for name in filenames:
            print("Loading composition: " + name + " ... ", end="")
            try:
                filepath = path + "/" + name
                equal_class = Composition.from_file(filepath)
                right_hand = equal_class.right_hand
                left_hand = equal_class.left_hand
                left_hand.elements.insert(0, Element(MessageType.meta, 0, std_velocity))
                left_hand.elements.insert(-1, Element(MessageType.meta, 1, std_velocity))
                for i in range(-5, 7):
                    treble_sequences.append(right_hand.transpose(i).to_neuron_representation())
                    bass_sequences.append(left_hand.transpose(i).to_neuron_representation())
                print("Done!")
            except Exception as e:
                print(e)

    target_sequences = np.zeros((len(bass_sequences), max(len(sequence) for sequence in bass_sequences), VOCAB_SIZE),
                      dtype="float32")
    for i, sequence in enumerate(bass_sequences):
        _, output = split(sequence)
        for j, message in enumerate(output):
            target_sequences[i, j, message] = 1.

    treble_sequences = K.preprocessing.sequence.pad_sequences(treble_sequences, padding="post")
    bass_sequences = K.preprocessing.sequence.pad_sequences(bass_sequences, padding="post")

    # treble_sequences = tf.data.Dataset.from_tensor_slices(treble_sequences).shuffle(10000).batch(batch_size, drop_remainder=True)
    # bass_sequences = tf.data.Dataset.from_tensor_slices(bass_sequences).shuffle(10000).batch(batch_size, drop_remainder=True)
    # target_sequences = tf.data.Dataset.from_tensor_slices(target_sequences).shuffle(10000).batch(batch_size, drop_remainder=True)

    # padded_sequences = K.preprocessing.sequence.pad_sequences(sequences, padding="post")
    # dataset = tf.data.Dataset.from_tensor_slices(padded_sequences)
    # dataset_split = dataset.map(split)
    # dataset_batches = dataset_split.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    return treble_sequences, bass_sequences, target_sequences


if __name__ == "__main__":
    shizzle()
