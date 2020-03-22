from mido import MidiFile, MetaMessage
from os import walk
from src.MusicElements import *


def main():
    for (dirpath, dirnames, filenames) in walk("../res"):
        for name in filenames:
            print(name)
            midi_file = MidiFile(dirpath + "/" + name)
            #sequence = mu.Sequence.from_midi_file(midi_file)
            #print(Sequence.average_complexity(sequence.split(sequence.numerator, sequence.denominator)))


def print_midi_file(midi_file: MidiFile, amount=-1) -> None:
    print("Start Midi File")
    print("Ticks per beat: " + str(midi_file.ticks_per_beat) + ", Type: " + str(midi_file.type))
    for j, track in enumerate(midi_file.tracks):
        for i, message in enumerate(track):
            if not message.is_meta: continue
            #if message.type == "set_tempo" or message.type == "end_of_track" or message.type == "midi_port": continue
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

if __name__ == '__main__':
    print_midi_file(MidiFile("../res/beethoven_op27_mo3.mid"))