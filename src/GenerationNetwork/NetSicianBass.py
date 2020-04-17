from __future__ import print_function

import keras as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Dropout
from mido import MidiFile

from src.Utility import *
from src.MusicElements import *
import numpy as np
import os

EPOCHS = 30
BATCH_SIZE = 8

SAVE_PATH = "../../out/net/bass/{complexity}"
LOAD_PATH = "../../out/lib/{complexity}"
CHECKPOINT_NAME = "cp_{epoch}"
MODEL_NAME = "model.h5"

VOCAB_SIZE = 203
NEURON_LIST = (1024, 1024, 1024)
DROPOUT = 0.2
EMBEDDING_DIM = 16


# TENSORFLOW


def build_models(neuron_list=NEURON_LIST):
    # Encoder Model
    enc_layer_input = Input(shape=(None,), name="Enc_Input")
    enc_layer_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True, name="Enc_Embedding")
    enc_layer_hidden_0 = LSTM(neuron_list[0], return_sequences=True, return_state=True, name="Enc_Hidden_0")
    enc_layer_dropout_0 = Dropout(DROPOUT)
    enc_layer_hidden_1 = LSTM(neuron_list[1], return_sequences=True, return_state=True, name="Enc_Hidden_1")
    enc_layer_dropout_1 = Dropout(DROPOUT)
    enc_layer_hidden_2 = LSTM(neuron_list[-1], return_state=True, name="Enc_Hidden_2")

    # Apply Layers
    enc_output_embedding = enc_layer_embedding(enc_layer_input)
    enc_output_hidden_0, enc_h0, enc_c0 = enc_layer_hidden_0(enc_output_embedding)
    enc_output_dropout_0 = enc_layer_dropout_0(enc_output_hidden_0)
    enc_output_hidden_1, enc_h1, enc_c1 = enc_layer_hidden_1(enc_output_dropout_0)
    enc_output_dropout_1 = enc_layer_dropout_1(enc_output_hidden_1)
    enc_output_hidden_2, enc_h2, enc_c2 = enc_layer_hidden_2(enc_output_dropout_1)

    # Save States
    enc_states = [enc_h0, enc_c0, enc_h1, enc_c1, enc_h2, enc_c2]

    # Decoder Model
    dec_layer_input = Input(shape=(None,), name="Dec_Input")
    dec_layer_embedding = Embedding(VOCAB_SIZE, neuron_list[0], mask_zero=True, name="Dec_Embedding")
    dec_layer_hidden_0 = LSTM(neuron_list[0], return_sequences=True, return_state=True, name="Dec_Hidden_0")
    dec_layer_dropout_0 = Dropout(DROPOUT)
    dec_layer_hidden_1 = LSTM(neuron_list[1], return_sequences=True, return_state=True, name="Dec_Hidden_1")
    dec_layer_dropout_1 = Dropout(DROPOUT)
    dec_layer_hidden_2 = LSTM(neuron_list[-1], return_sequences=True, return_state=True, name="Dec_Hidden_2")
    dec_layer_dropout_2 = Dropout(DROPOUT)
    dec_layer_output = Dense(VOCAB_SIZE, activation="softmax", name="Dec_Output")

    # Apply Layers
    dec_output_embedding = dec_layer_embedding(dec_layer_input)
    dec_output_hidden_0, _, _ = dec_layer_hidden_0(dec_output_embedding, initial_state=enc_states[0:2])
    dec_output_dropout_0 = dec_layer_dropout_0(dec_output_hidden_0)
    dec_output_hidden_1, _, _ = dec_layer_hidden_1(dec_output_dropout_0, initial_state=enc_states[2:4])
    dec_output_dropout_1 = dec_layer_dropout_1(dec_output_hidden_1)
    dec_output_hidden_2, _, _ = dec_layer_hidden_2(dec_output_dropout_1, initial_state=enc_states[4:6])
    dec_output_dropout_2 = dec_layer_dropout_2(dec_output_hidden_2)
    dec_output_output = dec_layer_output(dec_output_dropout_2)

    # Build Model
    training_model = Model(inputs=[enc_layer_input, dec_layer_input], outputs=dec_output_output, name="Training_Model")

    # =========
    # Inference
    # =========

    # Encoder
    ienc_model = Model(inputs=enc_layer_input, outputs=enc_states, name="Inference_Encoder_Model")

    # Decoder Model
    idec_layer_input_h0 = Input(shape=(neuron_list[0],), name="IDec_Input_h0")
    idec_layer_input_c0 = Input(shape=(neuron_list[0],), name="IDec_Input_c0")
    idec_layer_input_h1 = Input(shape=(neuron_list[1],), name="IDec_Input_h1")
    idec_layer_input_c1 = Input(shape=(neuron_list[1],), name="IDec_Input_c1")
    idec_layer_input_h2 = Input(shape=(neuron_list[2],), name="IDec_Input_h2")
    idec_layer_input_c2 = Input(shape=(neuron_list[2],), name="IDec_Input_c2")
    idec_states_input = [idec_layer_input_h0, idec_layer_input_c0, idec_layer_input_h1, idec_layer_input_c1,
                         idec_layer_input_h2, idec_layer_input_c2]

    # Apply Layers
    idec_output_embedding = dec_layer_embedding(dec_layer_input)
    idec_output_hidden_0, idec_h0, idec_c0 = dec_layer_hidden_0(idec_output_embedding,
                                                                initial_state=idec_states_input[0:2])
    idec_output_dropout_0 = dec_layer_dropout_0(idec_output_hidden_0)
    idec_output_hidden_1, idec_h1, idec_c1 = dec_layer_hidden_1(idec_output_dropout_0,
                                                                initial_state=idec_states_input[2:4])
    dec_output_dropout_1 = dec_layer_dropout_1(idec_output_hidden_1)
    idec_output_hidden_2, idec_h2, idec_c2 = dec_layer_hidden_1(dec_output_dropout_1,
                                                                initial_state=idec_states_input[4:6])
    dec_output_dropout_2 = dec_layer_dropout_2(idec_output_hidden_2)
    idec_output_output = dec_layer_output(dec_output_dropout_2)

    # Save States
    idec_states = [idec_h0, idec_c0, idec_h1, idec_c1, idec_h2, idec_c2]

    # Build Model
    idec_model = Model(inputs=[dec_layer_input] + idec_states_input, outputs=[idec_output_output] + idec_states,
                       name="Interference_Decoder_Model")

    return training_model, ienc_model, idec_model


def split(chunk):
    seq_input = chunk[:-1]
    seq_output = chunk[1:]
    return seq_input, seq_output


def setup_tensorflow():
    # Limit memory consumption
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Set log level
    tf.get_logger().setLevel("ERROR")


# CLASS


def load_data(complexity):
    path = LOAD_PATH.format(complexity=str(complexity).lower())

    treble_sequences = []
    bass_sequences = []

    for (dirpath, dirnames, filenames) in os.walk(path):
        for name in filenames:
            print("Loading composition: " + name + " ... ", end="")
            try:
                filepath = path + "/" + name
                equal_class = Composition.from_midi_file(MidiFile(filepath))[0]
                equal_class.preprocess()

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

    sequences = list(zip(treble_sequences, bass_sequences))
    random.shuffle(sequences)

    treble_sequences, bass_sequences = zip(*sequences)

    target_sequences = np.zeros((len(bass_sequences), max(len(sequence) for sequence in bass_sequences), VOCAB_SIZE),
                                dtype="float32")
    for i, sequence in enumerate(bass_sequences):
        _, output = split(sequence)
        for j, message in enumerate(output):
            target_sequences[i, j, message] = 1.

    treble_sequences = K.preprocessing.sequence.pad_sequences(treble_sequences, padding="post")
    bass_sequences = K.preprocessing.sequence.pad_sequences(bass_sequences, padding="post")

    return treble_sequences, bass_sequences, target_sequences


def train(complexity):
    # Load data
    treble_sequences, bass_sequences, target_sequences = load_data(complexity)

    # Build models
    models = build_models()
    training_model = models[0]
    training_model.summary()
    print()

    # Set Save Path
    save_path = SAVE_PATH.format(complexity=str(complexity).lower())

    callback = K.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, CHECKPOINT_NAME), save_weights_only=True)

    # Run training
    training_model.compile(optimizer="adam", loss="categorical_crossentropy",
                           metrics=["accuracy"])

    training_model.fit([treble_sequences, bass_sequences], target_sequences,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       callbacks=[callback],
                       verbose=1)
    # Save model
    training_model.save(os.path.join(save_path, MODEL_NAME))


def generate(input, tempmodels):
    # models = build_models()
    models = tempmodels

    training_model = models[0]
    encoder_model = models[1]
    decoder_model = models[2]

    states = encoder_model.predict(input)

    # Prepare output, start with start word
    output = np.zeros((1, VOCAB_SIZE))
    output[0, Constants.START_WORD] = 1.

    flag_stop = False

    elements = []

    while not flag_stop:
        output_token, h0, c0, h1, c1, h2, c2 = decoder_model.predict([output] + states)

        sampled_token = np.argmax(output_token[0, -1, :])
        if sampled_token == Constants.PADDING or sampled_token == Constants.START_WORD:
            print("Start or pad")
            continue
        if sampled_token == Constants.END_WORD:
            flag_stop = True
            continue

        try:
            sampled_element = Element.from_neuron_representation(sampled_token)
        except Exceptions.InvalidRepresentation:
            print("Invalid representation of {token}".format(token=sampled_token))
            output = np.zeros((1, VOCAB_SIZE))
            output[0, 0] = 1.
            states = [h0, c0, h1, c1, h2, c2]
            continue

        elements.append(sampled_element)

        output = np.zeros((1, VOCAB_SIZE))
        output[0, sampled_element.to_neuron_representation()] = 1.

        states = [h0, c0, h1, c1, h2, c2]
        flag_stop = True

    # Encode input


def generate_stuff(stuff):
    models = build_models()
    training_model = models[0]
    training_model.load_weights(os.path.join(SAVE_PATH, MODEL_NAME))

    generate(stuff, models)


if __name__ == "__main__":
    setup_tensorflow()

    # Train Model
    train(Complexity.MEDIUM)
