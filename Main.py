from mido import MidiFile, MetaMessage

import Music as mu
from Music import Sequence, MessageType, Element


def main():
    # midiFile = MidiFile("res/bach_op27_r.mid")
    midiFile = MidiFile("res/bach_op27_r.mid")

    bach = Sequence.fromMidiFile(midiFile)
    created = bach.toMidiFile()

    created.save("res/created.mid")


def test():
    musical = mu.Musical.fromMidiFiles(MidiFile("res/beethoven_op27_r.mid"), MidiFile("res/beethoven_op27_l.mid"))
    musical.toMidiFile().save("out/beethoven_op27.mid")


if __name__ == '__main__':
    # main()
    # test()
    test()
