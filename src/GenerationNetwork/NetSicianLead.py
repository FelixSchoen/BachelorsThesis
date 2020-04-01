from __future__ import annotations
import queue
import os
import tensorflow as tf
import random
from src.MusicElements import *
from src.GenerationNetwork import *
from src.Utility import *
from threading import Thread
from mido import MidiFile

EPOCHS = 5
BATCH_SIZE = 16
BUFFER_SIZE = 2048

CHECKPOINT_PATH = "out/net/lead"
CHECKPOINT_NAME = "cp_{epoch}"

VOCAB_SIZE = 200
NEURON_LIST = (1024, 512, 512)
DROPOUT = 0.25
EMBEDDING_DIM = 32


def build_model(neuron_list=NEURON_LIST, batch_size=BATCH_SIZE, dropout=DROPOUT, embedding_dim=EMBEDDING_DIM):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE + 1,
                                  embedding_dim,
                                  batch_input_shape=[batch_size, None],
                                  mask_zero=True),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(neuron_list[0],
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(neuron_list[1],
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(neuron_list[2]),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(VOCAB_SIZE + 1)
    ])
    return model


def split(chunk):
    seq_input = chunk[:-1]
    seq_output = chunk[1:]
    return seq_input, seq_output


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def setup_tensorflow():
    # Limit memory consumption
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Set log level
    tf.get_logger().setLevel("ERROR")


def load_data():
    filepaths = []

    for (dirpath, dirnames, filenames) in os.walk("../../res/midi"):
        for name in filenames:
            filepath = dirpath + "/" + name
            filepaths.append(filepath)

    sequences = []
    filepaths = util_remove_elements(filepaths, 0.95)

    for filepath in filepaths:
        midi_file = MidiFile(filepath)
        try:
            compositions = Composition.from_midi_file(midi_file)
        except Exception:
            print("Error loading composition: " + filepath)
            continue

        for composition in compositions:
            # At this time only accept 4/4 compositions
            if composition.numerator / composition.denominator != 4 / 4:
                continue

            bars = composition.split_to_bars()
            equal_complexity_classes = Composition.stitch_to_equal_difficulty_classes(bars, track_identifier=RIGHT_HAND)
            for equal_complexity_class in equal_complexity_classes:
                for i in range(-5, 7):
                    sequences.append(equal_complexity_class.right_hand.transpose(i).to_neuron_representation())
        print("Loaded composition: " + filepath)

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")
    dataset = tf.data.Dataset.from_tensor_slices(padded_sequences)
    dataset_split = dataset.map(split)
    dataset_batches = dataset_split.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    return dataset_batches


def load_model():
    # Build model
    model = build_model()

    # Try to load existing weights
    try:
        model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_PATH))
    except AttributeError:
        print("Weights could not be loaded")

    # Compile model
    model.compile(optimizer="rmsprop", loss=loss)

    return model


def util_remove_elements(elements: list, percentage_to_drop: float) -> list:
    returned_list = []
    for element in elements:
        if not random.random() < percentage_to_drop:
            returned_list.append(element)
    return returned_list


if __name__ == "__main__":
    setup_tensorflow()
    data = load_data()
    model = load_model()

    model.summary()
    print()
    print(data)

    model.fit(data, epochs=EPOCHS, verbose=1)
