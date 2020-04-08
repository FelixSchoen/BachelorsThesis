from mido import MidiFile, MetaMessage
from os import walk
from src.MusicElements import *
from src.Utility import *
from concurrent.futures import ThreadPoolExecutor


def main():
    for (dirpath, dirnames, filenames) in walk("../res"):
        for name in filenames:
            print(name)
            midi_file = MidiFile(dirpath + "/" + name)
            # sequence = mu.Sequence.from_midi_file(midi_file)
            # print(Sequence.average_complexity(sequence.split(sequence.numerator, sequence.denominator)))


def print_midi_file(midi_file: MidiFile, amount=-1) -> None:
    print("Start Midi File")
    print("Ticks per beat: " + str(midi_file.ticks_per_beat) + ", Type: " + str(midi_file.type))
    for j, track in enumerate(midi_file.tracks):
        for i, message in enumerate(track):
            if not message.is_meta: continue
            # if message.type == "set_tempo" or message.type == "end_of_track" or message.type == "midi_port": continue
            if i >= amount != -1:
                break
            print("Track " + str(j) + ": " + str(message))
    print("End Midi File")


def temp():
    midi_file = MidiFile("../res/beethoven_op27.mid")
    seq = SequenceRelative.from_midi_track(midi_file.tracks[2])
    abs = seq.to_absolute_sequence()
    print(abs)
    abs.quantize()
    print(abs)
    seq = abs.to_relative_sequence()
    print(seq)
    seq.adjust()
    print(seq)

    track = seq.to_absolute_sequence().quantize().to_relative_sequence().adjust().to_midi_track()
    print(SequenceRelative.from_midi_track(track))
    midi_file.tracks[2] = track
    midi_file.save("../out/generated.mid")


def judge_difficulty_stitch_and_persist(name, midi_file):
    try:
        compositions = Composition.from_midi_file(midi_file)
        for composition in compositions:
            if composition.denominator != 4:
                continue
            equal_classes = Composition.stitch_equal_complexity(composition.split_to_bars(), Constants.RIGHT_HAND)
            for i, equal_class in enumerate(equal_classes):
                if equal_class.final_complexity == Complexity.EASY:
                    equal_class.to_midi_file().save("../out/lib/easy/" + name + "-" + str(i) + ".mid")
                elif equal_class.final_complexity == Complexity.MEDIUM:
                    equal_class.to_midi_file().save("../out/lib/medium/" + name + "-" + str(i) + ".mid")
                else:
                    equal_class.to_midi_file().save("../out/lib/hard/" + name + "-" + str(i) + ".mid")

    except Exception as e:
        print(name, e)


def load_midi_files_and_persist():
    executor = ThreadPoolExecutor()

    directories = []
    for (dirpath, dirnames, filenames) in walk("../res/midi"):
        for name in filenames:
            directories.append((dirpath + "/" + name, name))
    # directories = Util.util_remove_elements(directories, 1)

    for pairing in directories:
        midi_file = MidiFile(pairing[0])
        executor.submit(judge_difficulty_stitch_and_persist, pairing[1][:-4], midi_file)


if __name__ == '__main__':
    load_midi_files_and_persist()
