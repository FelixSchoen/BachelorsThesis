from __future__ import annotations
import queue
import os
import tensorflow as tf
from src.MusicElements import SequenceRelative, Element
from threading import Thread
from mido import MidiFile


class NetSicianLead:
    BATCH_SIZE = 12  # For all transponations
    MAX_QUEUE_SIZE = 5
    CHECKPOINT_PATH = "out/net/lead"
    CHECKPOINT_NAMES = "cp_{epoch}"
    CHECKPOINT = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAMES)
    CALLBACK = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT, save_weights_only=True)

    def __init__(self) -> None:
        self.flag_active = True
        self.queue = queue.Queue(self.MAX_QUEUE_SIZE)
        self.model = None

    def start(self):
        self.setup()
        elements = []
        while self.flag_active:
            try:
                sequences = (self.queue.get(True, 3))
                entrypoint(self.model, self.CALLBACK, sequences)
            except queue.Empty:
                self.flag_active = False
                if len(elements) > 0:
                    entrypoint(self.model, self.CALLBACK, elements)

    def setup(self):
        # Limit memory, otherwise crashes all the time
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.model = build_model(256, [256, 256, 256, 256, 256])
        try:
            self.model.load_weights(tf.train.latest_checkpoint(self.CHECKPOINT_PATH))
        except AttributeError:
            print("Could not load weights")
        self.model.compile(optimizer="adam", loss=loss)

    def add_sequence(self, sequence: list[SequenceRelative]):
        self.queue.put(sequence, block=True)

    def deactivate(self):
        self.flag_active = False

    def run(self):
        t1 = Thread(target=self.start)
        t1.start()


BUFFER_SIZE = 256
VOCAB_SIZE = 200
EPOCHS = 10


def build_model(embedding_dim, rnn_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim,
                                  batch_input_shape=[1, None]),
        tf.keras.layers.LSTM(rnn_units[0],
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(rnn_units[1],
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(rnn_units[2],
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(rnn_units[3],
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(rnn_units[4],
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(VOCAB_SIZE)
    ])
    return model


def split(chunk):
    seq_input = chunk[:-1]
    seq_ouptut = chunk[1:]
    return seq_input, seq_ouptut


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def entrypoint(model, callback, sequences: list[SequenceRelative]):
    tensor_list = []

    for sequence in sequences:
        tensor = tf.convert_to_tensor(sequence.to_neuron_representation())
        tensor_list.append(tensor)

    dataset = tf.data.Dataset.from_generator(lambda: tensor_list, tf.int32, output_shapes=[None])
    dataset = dataset.map(split)
    data = dataset.shuffle(BUFFER_SIZE).batch(1, drop_remainder=True)

    model.fit(data, epochs=EPOCHS, callbacks=[callback])


def generate(model, start, num, temp):
    generated = []
    values = start
    values = tf.expand_dims(values, 0)

    model.reset_states()
    for i in range(num):
        predictions = model(values)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temp
        predicted = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        values = tf.expand_dims([predicted], 0)

        generated.append(predicted)

    return generated


if __name__ == "__main__":
    model = build_model(256, [256, 256, 256, 256, 256])

    model.load_weights(tf.train.latest_checkpoint("../../out/net/lead"))
    model.build(tf.TensorShape([1, None]))

    generated = generate(model, [144], 500, 1)
    final = []
    for num in generated:
        final.append(Element.from_neuron_representation(num))

    print(final)
    seq = SequenceRelative()
    seq.elements = final
    seq.adjust()
    print(seq)
    file = MidiFile()
    file.tracks.append(seq.to_midi_track())
    file.save("out.mid")
