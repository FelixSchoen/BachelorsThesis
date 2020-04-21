from __future__ import annotations
import os
import tensorflow as tf
import random
from src.MusicElements import *
from src.Utility import *
from mido import MidiFile

EPOCHS = 30
BATCH_SIZE = 64
BUFFER_SIZE = 4096

SAVE_PATH = "../../out/net/treble/{complexity}"
LOAD_PATH = "../../out/lib/{complexity}"
CHECKPOINT_NAME = "cp_{epoch}"
MODEL_NAME = "model.h5"

# 200 Values + Padding Size
VOCAB_SIZE = 201
NEURON_LIST = (1024, 1024, 1024)
DROPOUT = 0.2
EMBEDDING_DIM = 32


# BEGIN TENSORFLOW CONFIGURATION


def build_model(neuron_list=NEURON_LIST, batch_size=BATCH_SIZE, dropout=DROPOUT, embedding_dim=EMBEDDING_DIM):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE,
                                  embedding_dim,
                                  batch_input_shape=[batch_size, None],
                                  mask_zero=True),
        tf.keras.layers.LSTM(neuron_list[0],
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(neuron_list[1],
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(neuron_list[2],
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(VOCAB_SIZE)
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


# BEGIN CLASS FUNCTIONS


def load_data(complexity, batch_size=BATCH_SIZE):
    path = LOAD_PATH.format(complexity=str(complexity).lower())

    sequences = []

    for (dirpath, dirnames, filenames) in os.walk(path):
        for name in filenames:
            print("Loading composition: " + name + " ... ", end="")
            try:
                filepath = path + "/" + name
                equal_class = Composition.from_midi_file(MidiFile(filepath))[0]
                equal_class.preprocess()
                for i in range(-5, 7):
                    sequences.append(equal_class.right_hand.transpose(i).to_neuron_representation())
                print("Done!")
            except Exception as e:
                print(e)

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")
    dataset = tf.data.Dataset.from_tensor_slices(padded_sequences)
    dataset_split = dataset.map(split)
    dataset_batches = dataset_split.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    return dataset_batches


def train(complexity):
    # Load data
    data = load_data(complexity, BATCH_SIZE)

    # Build model
    model = build_model(neuron_list=NEURON_LIST, batch_size=BATCH_SIZE)

    model.summary()
    print()

    # Set Save Path
    save_path = SAVE_PATH.format(complexity=str(complexity).lower())

    callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, CHECKPOINT_NAME),
                                                  save_weights_only=True)

    # Compile model
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    model.fit(data, epochs=EPOCHS, callbacks=[callback], verbose=1)

    model.save_weights(os.path.join(save_path, MODEL_NAME))


def generate_bars(model, start_sequence, bars, temperature) -> SequenceRelative:
    numerator = 4
    denominator = 4

    # Calculate maximum time for bars
    max_time = 4 * int(numerator / denominator) * internal_ticks * bars
    wait_time = 0

    generated = []

    # Append start sequence and add to wait time
    for value in start_sequence:
        element = Element.from_neuron_representation(value)
        generated.append(element)
        if element.message_type == MessageType.wait:
            wait_time += element.value

    # Convert input values
    input_values = start_sequence
    input_values = tf.expand_dims(input_values, 0)

    model.reset_states()
    while wait_time < max_time:
        predictions = model(input_values)
        # List of batches with size 1 to list of generated elements
        predictions = tf.squeeze(predictions, 0)

        # Truncated sampling
        predictions = predictions / temperature
        predicted = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        if predicted == 0:
            # Padding value
            continue

        # Last element to predicted elements
        input_values = tf.expand_dims([predicted], 0)

        # Add element to generated elements and add to wait time
        element = Element.from_neuron_representation(predicted)
        if element.message_type == MessageType.wait:
            wait_time += element.value
        generated.append(element)

    sequence = SequenceRelative(numerator, denominator)
    sequence.elements = generated
    sequence = sequence.to_absolute_sequence().cutoff(force=True).to_relative_sequence().adjust()
    return sequence.split(max_time)[0]


def generate(complexity, bars, start_sequence=[140], checkpoint=-1, temp=1.0) -> SequenceRelative:
    # Load model with batch size of 1
    model = build_model(batch_size=1)

    # Set Save Path
    save_path = SAVE_PATH.format(complexity=str(complexity).lower())

    if checkpoint == -1:
        model.load_weights(os.path.join(save_path, MODEL_NAME))
    else:
        model.load_weights(save_path + "\cp_" + str(checkpoint))

    model.build(tf.TensorShape([1, None]))

    seq = generate_bars(model, start_sequence, bars, temp)
    seq = seq.to_absolute_sequence().quantize().to_relative_sequence().adjust()

    return seq


if __name__ == "__main__":
    setup_tensorflow()

    # Train Model
    # train(Complexity.MEDIUM)

    # Generate Sequence
    sequence = generate(Complexity.EASY, 8, temp=1.0)

    # Generate Midi File
    midi_file = MidiFile()
    midi_file.tracks.append(sequence.to_midi_track())
    midi_file.save("../../out/gen/treble_easy.mid")
