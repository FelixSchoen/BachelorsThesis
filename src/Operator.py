from src.Utility import *
from src.MusicElements import *
from src.GenerationNetwork import *
from src.Utility import *
from mido import MidiFile
import os
import random


def load_and_train_lead():
    lead = NetSicianLead()

    filepaths = []

    for (dirpath, dirnames, filenames) in os.walk("./res/midi"):
        for name in filenames:
            filepath = dirpath + "/" + name
            filepaths.append(filepath)

    sequences = []

    for filepath in filepaths:
        midi_file = MidiFile(filepath)

        compositions = Composition.from_midi_file(midi_file)

        lead.run()

        for composition in compositions:
            print(filepath)
            if composition.numerator / composition.denominator == 4 / 4:
                bars = composition.split_to_bars()
                stitched = Composition.stitch_to_equal_difficulty_classes(bars, track_identifier=RIGHT_HAND)
                for i in range(-5, 7):
                    for difficulty_class in stitched:
                        sequences.append(difficulty_class.transpose(i).right_hand)
                    random.shuffle(sequences)
                    lead.add_sequence(sequences)
                    sequences = []

