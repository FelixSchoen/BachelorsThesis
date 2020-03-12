from mido import MidiFile

from Music import Musical, MessageType, Element


def main():
    midiFile = MidiFile("res/bach_op27_r.mid")

    Musical.fromMidiFile(midiFile)


if __name__ == '__main__':
    main()
