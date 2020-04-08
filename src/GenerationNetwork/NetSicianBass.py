from __future__ import print_function

import keras as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from src.Utility import *
from src.MusicElements import *
import numpy as np
import os

EPOCHS = 30
BATCH_SIZE = 256
BUFFER_SIZE = 4096

VOCAB_SIZE = 203
NEURON_LIST = (64, 64)
DROPOUT = 0.2
EMBEDDING_DIM = 16


def build_models(neuron_list=NEURON_LIST):
    # Encoder Model
    enc_layer_input = Input(shape=(None,), name="Enc_Input")
    enc_layer_embedding = Embedding(VOCAB_SIZE, neuron_list[0], mask_zero=True, name="Enc_Embedding")
    enc_layer_hidden_0 = LSTM(neuron_list[-1], return_state=True, name="Enc_Hidden_0")

    # Apply Layers
    enc_output_embedding = enc_layer_embedding(enc_layer_input)
    enc_output_hidden_0, enc_h0, enc_c0 = enc_layer_hidden_0(enc_output_embedding)

    # States
    enc_states = [enc_h0, enc_c0]

    # Decoder Model
    dec_layer_input = Input(shape=(None,), name="Dec_Input")
    dec_layer_embedding = Embedding(VOCAB_SIZE, neuron_list[0], mask_zero=True, name="Dec_Embedding")
    dec_layer_hidden_0 = LSTM(neuron_list[-1], return_sequences=True, return_state=True, name="Dec_Hidden_0")
    dec_layer_output = Dense(VOCAB_SIZE, activation="softmax", name="Dec_Output")

    # Apply Layers
    dec_output_embedding = dec_layer_embedding(dec_layer_input)
    dec_output_hidden_0, dec_h0, dec_c0 = dec_layer_hidden_0(dec_output_embedding, initial_state=enc_states[0:2])
    dec_output_output = dec_layer_output(dec_output_hidden_0)

    training_model = Model(inputs=[enc_layer_input, dec_layer_input], outputs=dec_output_output, name="Training_Model")
    training_model.summary()

    # =========
    # Inference
    # =========

    # Encoder
    ienc_model = Model(inputs=enc_layer_input, outputs=enc_states, name="Inference_Encoder_Model")
    ienc_model.summary()

    # Decoder Model
    idec_layer_input_h0 = Input(shape=(neuron_list[0],), name="IDec_Input_h0")
    idec_layer_input_c0 = Input(shape=(neuron_list[0],), name="IDec_Input_c0")
    idec_states_input = [idec_layer_input_h0, idec_layer_input_c0]

    # Apply Layers
    idec_output_embedding = dec_layer_embedding(dec_layer_input)
    idec_output_hidden_0, idec_h0, idec_c0 = dec_layer_hidden_0(idec_output_embedding,
                                                                initial_state=idec_states_input[0:2])
    idec_output_output = dec_layer_output(idec_output_hidden_0)

    idec_states = [idec_h0, idec_c0]

    idec_model = Model(inputs=[dec_layer_input] + idec_states_input, outputs=[idec_output_output] + idec_states, name="Interference_Decoder_Model")
    idec_model.summary()

    return training_model, ienc_model, idec_model


def shizzle():
    batch_size = 64  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    num_samples = 10000  # Number of samples to train on.

    treble_sequences, bass_sequences, target_sequences = load_pickle_data(Complexity.MEDIUM, batch_size)

    models = build_models()[0]
    # training_model = models[0]
    # training_model.summary()
    #
    # # Run training
    # training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    #                        metrics=['accuracy'])
    # model.fit([treble_sequences, bass_sequences], target_sequences,
    #           batch_size=BATCH_SIZE,
    #           epochs=epochs)
    # # Save model
    # model.save('s2s.h5')


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

    # padded_sequences = K.preprocessing.sequence.pad_sequences(sequences, padding="post")
    # dataset = tf.data.Dataset.from_tensor_slices(padded_sequences)
    # dataset_split = dataset.map(split)
    # dataset_batches = dataset_split.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    return treble_sequences, bass_sequences, target_sequences


if __name__ == "__main__":
    shizzle()
