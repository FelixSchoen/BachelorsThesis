from src.MusicElements import *
from mido import MidiFile
import tensorflow as tf
import os
import numpy as np
import random


def split(chunk):
    input = chunk[:-1]
    output = chunk[1:]
    return input, output


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

def test(list):
    return list


def entrypoint():
    compositions = Composition.from_midi_file(MidiFile("../../res/beethoven_op27_mo3.mid"))

    tensor_list = []
    for i, bar in enumerate(compositions[0].split_to_bars()):
        if i < 10:
            print(i, len(bar.right_hand.to_neuron_representation()))
        tensor = tf.convert_to_tensor(bar.right_hand.to_neuron_representation())
        tensor_list.append(tensor)

    dataset = tf.data.Dataset.from_generator(lambda: tensor_list, tf.int32, output_shapes=[None])
    dataset = dataset.map(split)

    data = dataset.shuffle(10000).batch(1, drop_remainder=True)
    for x in data:
        print(x)

    model = build_model(200, 256, 1024, 1)
    model.summary()

    for input_example_batch, target_example_batch in data.take(1):
        example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape
        print(len(example_batch_predictions))
        print(len(example_batch_predictions[0]))
        print(len(example_batch_predictions[0][0]))

    model.compile(optimizer="adam", loss=loss)
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    history = model.fit(data, epochs=5, callbacks=[checkpoint_callback])

def generate(model, start, num):
    generated = []
    temp = 1.0
    values = start
    values = tf.expand_dims(values, 0)

    model.reset_states()
    for i in range(num):
        predictions = model(values)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temp
        predicted = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        start = tf.expand_dims([predicted], 0)

        generated.append(predicted)

    return generated

def pentrypoint():
    model = build_model(200, 256, 1024, 1)
    model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
    model.build(tf.TensorShape([1, None]))

    generated = generate(model, [100], 20)
    final = []
    for num in generated:
        final.append(Element.from_neuron_representation(num))

    print(final)
    seq = SequenceRelative()
    seq.elements = final
    seq.adjust()
    file = MidiFile()
    file.tracks.append(seq.to_midi_track())
    file.save("save.mid")

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    entrypoint()
    pentrypoint()
