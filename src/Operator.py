from src.Utility import *
from src.MusicElements import *
from src.GenerationNetwork import *
from src.Utility import *
from mido import MidiFile
import os
import random
import json

def load_files():
    filepaths = []

    for (dirpath, dirnames, filenames) in os.walk("./res/midi"):
        for name in filenames:
            filepath = dirpath + "/" + name
            filepaths.append(filepath)

    random.shuffle(filepaths)

    with open("files.json", "w") as f:
        json.dump(filepaths, f)

def load_and_train_lead():
    lead = NetSicianLead()

    with open("files.json", "r") as f:
        filepaths = json.load(f)

    sequences = []

    while len(filepaths) > 0:
        filepath = filepaths.pop(0)
        midi_file = MidiFile(filepath)

        try:
            compositions = Composition.from_midi_file(midi_file)
        except Exception:
            print("ERROR: " + str(filepath))
            continue

        for composition in compositions:
            print(filepath)
            if composition.numerator / composition.denominator == 4 / 4:
                bars = composition.split_to_bars()
                stitched = Composition.stitch_to_equal_difficulty_classes(bars, track_identifier=RIGHT_HAND)
                transpose = list(range(-5, 7))
                # for difficulty_class in stitched:
                for i in transpose:
                    sequences.append(composition.transpose(i).right_hand)

        with open("files.json", "w") as f:
            json.dump(filepaths, f)

    lead.run()
    lead.add_sequence(sequences)

    lead.flag_active = False
