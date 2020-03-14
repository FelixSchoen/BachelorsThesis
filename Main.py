from mido import MidiFile, MetaMessage

import Music as mu
from Music import Sequence, MessageType, Element


def main():
    print("Lel")

def test():
    musical = mu.Musical.from_midi_file(MidiFile("res/beethoven_op27_r.mid"), MidiFile("res/beethoven_op27_l.mid"))
    print(musical.detect_scale())
    musical.transpose(-4)
    musical.to_midi_file().save("out/beethoven_op27n.mid")
    print(musical.detect_scale())


if __name__ == '__main__':
    # main()
    # test()
    test()
