from src.Utility import *
from src.MusicElements import *
from src.GenerationNetwork import *
from src.Utility import *
from mido import MidiFile
import os


def load_and_train_lead():
    lead = NetSicianLead()
    lead.run()

    for (dirpath, dirnames, filenames) in os.walk("./res/midi"):
        for name in filenames:
            print(name)
            midi_file = MidiFile(dirpath + "/" + name)
            compositions = Composition.from_midi_file(midi_file)
            sequences = []
            for composition in compositions:
                if composition.numerator / composition.denominator == 4 / 4:
                    bars = composition.split_to_bars()
                    stitched = Composition.stitch_to_equal_difficulty_classes(bars, track_identifier=RIGHT_HAND)
                    for i in range(-5, 7):
                        for difficulty_class in stitched:
                            sequences.append(difficulty_class.transpose(i).right_hand)
                        lead.add_sequence(sequences)
                        sequences = []