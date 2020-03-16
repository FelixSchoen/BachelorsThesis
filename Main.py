from mido import MidiFile, MetaMessage

import Music as mu
from Music import Sequence, MessageType, Element


def main():
    musical = mu.Musical.from_midi_file(MidiFile("res/beethoven_op27_r.mid"), MidiFile("res/beethoven_op27_l.mid")).to_midi_file()

def test():
    seq = mu.Sequence.from_midi_file(MidiFile("res/beethoven_op27_csmin_mo1_0.mid"))
    print(seq.detect_scale()[0])
    print(seq.detect_scale()[1])
    mu.Sequence.stitch_together(seq.split(seq.numerator, seq.denominator)).to_midi_file().save("out/stitched.mid")


if __name__ == '__main__':
    #main()
    test()
