from mido import MidiFile, MetaMessage
from os import walk
import Music as mu
from Music import Sequence, MessageType, Element


def main():
    for (dirpath, dirnames, filenames) in walk("res"):
        for name in filenames:
            print(name)
            midi_file = MidiFile(dirpath + "/" + name)
            sequence = mu.Sequence.from_midi_file(midi_file)
            print(Sequence.average_complexity(sequence.split(sequence.numerator, sequence.denominator)))


def test():
    seq = Sequence.from_midi_file(MidiFile("res/4-4/beethoven_op27_csmin_mo3_0.mid"))
    seq.to_midi_file().save("out/half.mid")
    hash = set()
    for element in seq.elements:
        if element.message_type == MessageType.wait:
            hash.add(element.value)
    print(sorted(hash))


def asdf():
    seq = mu.Sequence.from_midi_file(MidiFile("res/q/quantized.mid"))
    hash = set()
    for element in seq.elements:
        if element.message_type == MessageType.wait:
            hash.add(element.value)

    print(sorted(hash))

    hash.clear()
    for element in seq.elements:
        if element.message_type == MessageType.wait:
            hash.add(mu.Sequence.quantize_note(element.value))

    print(sorted(hash))

def quantest():
    list = [8, 10, 14]
    nl = set()
    for i in list:
        nl.add(mu.Sequence.quantize_note(i))
    print(sorted(nl))


if __name__ == '__main__':
    asdf()
